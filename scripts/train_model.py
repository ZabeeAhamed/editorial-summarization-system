import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW 
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class EditorialDataset(Dataset):
    
    def __init__(self, df, tokenizer, max_source_length=512, max_target_length=150):
        self.tokenizer = tokenizer
        self.articles = df['Article'].tolist()
        self.summaries = df['Summary'].tolist()
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
    
    def __len__(self):
        return len(self.articles)
    
    def __getitem__(self, idx):
        article = str(self.articles[idx])
        summary = str(self.summaries[idx])
        
        # T5 expects "summarize: " prefix
        source = f"summarize: {article}"
        
        # Tokenize source
        source_encoding = self.tokenizer(
            source,
            max_length=self.max_source_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize target
        target_encoding = self.tokenizer(
            summary,
            max_length=self.max_target_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        labels = target_encoding['input_ids']
        labels[labels == self.tokenizer.pad_token_id] = -100  
        
        return {
            'input_ids': source_encoding['input_ids'].squeeze(),
            'attention_mask': source_encoding['attention_mask'].squeeze(),
            'labels': labels.squeeze()
        }

class T5Trainer:
    #Fine-tuning T5
    def __init__(self, model_name='t5-small', device=None):
        self.model_name = model_name
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        print("="*80)
        print("EDITORIAL SUMMARIZATION - T5 FINE-TUNING")
        print("="*80)
        print(f"Author: ZabeeAhmed")
        print(f"Date: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
        print(f"Device: {self.device}")
        print("="*80 + "\n")
        
        # Load tokenizer and model
        print("Loading T5 model and tokenizer...")
        self.tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.model.to(self.device)
        print("Model loaded successfully!\n")
    
    def prepare_data(self, csv_path, train_split=0.8, val_split=0.1):
        """Load and split dataset"""
        print(f" Loading dataset from: {csv_path}")
        df = pd.read_csv(csv_path)
        print(f"   Total samples: {len(df)}\n")
        
        # Shuffle
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Split
        n = len(df)
        train_size = int(n * train_split)
        val_size = int(n * val_split)
        
        train_df = df[:train_size]
        val_df = df[train_size:train_size + val_size]
        test_df = df[train_size + val_size:]
        
        print(f"Data split:")
        print(f"Training: {len(train_df)} samples ({train_split*100:.0f}%)")
        print(f"Validation: {len(val_df)} samples ({val_split*100:.0f}%)")
        print(f"Test: {len(test_df)} samples ({(1-train_split-val_split)*100:.0f}%)\n")
        
        # Create datasets
        train_dataset = EditorialDataset(train_df, self.tokenizer)
        val_dataset = EditorialDataset(val_df, self.tokenizer)
        test_dataset = EditorialDataset(test_df, self.tokenizer)
        
        return train_dataset, val_dataset, test_dataset, test_df
    
    def train(
        self,
        train_dataset,
        val_dataset,
        output_dir='models/finetuned',
        batch_size=4,
        num_epochs=5,
        learning_rate=3e-4,
        warmup_steps=100
    ):
        """Fine-tune the model"""
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        total_steps = len(train_loader) * num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        print("Training Configuration:")
        print(f"Batch size: {batch_size}")
        print(f"Epochs: {num_epochs}")
        print(f"Learning rate: {learning_rate}")
        print(f"Total steps: {total_steps}")
        print(f"Warmup steps: {warmup_steps}\n")
        
        # Training loop
        print("Starting training...\n")
        
        training_stats = []
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            print(f"{'='*80}")
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"{'='*80}")
            
            # Training phase
            self.model.train()
            train_loss = 0
            
            progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
            for batch in progress_bar:
                # Move to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                train_loss += loss.item()
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                # Update progress bar
                progress_bar.set_postfix({'loss': loss.item()})
            
            avg_train_loss = train_loss / len(train_loader)
            
            # Validation phase
            self.model.eval()
            val_loss = 0
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}"):
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    
                    val_loss += outputs.loss.item()
            
            avg_val_loss = val_loss / len(val_loader)
            
            # Log stats
            print(f"\n Epoch {epoch + 1} Results:")
            print(f"   Training Loss: {avg_train_loss:.4f}")
            print(f"   Validation Loss: {avg_val_loss:.4f}\n")
            
            training_stats.append({
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss
            })
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                print(f"Saving best model (val_loss: {avg_val_loss:.4f})...")
                self.model.save_pretrained(output_dir)
                self.tokenizer.save_pretrained(output_dir)
                print(f"   Saved to: {output_dir}\n")
        
        # Save training stats
        stats_path = output_dir / 'training_stats.json'
        with open(stats_path, 'w') as f:
            json.dump(training_stats, f, indent=2)
        
        print(f"{'='*80}")
        print("TRAINING COMPLETE!")
        print(f"{'='*80}")
        print(f" Best model saved to: {output_dir}")
        print(f" Training stats saved to: {stats_path}")
        print(f"   Best validation loss: {best_val_loss:.4f}\n")
        
        return training_stats
    
    def generate_summary(self, text, max_length=150):
        """Generate summary for a single article"""
        self.model.eval()
        
        input_text = f"summarize: {text}"
        inputs = self.tokenizer(
            input_text,
            max_length=512,
            truncation=True,
            return_tensors='pt'
        ).to(self.device)
        
        with torch.no_grad():
            summary_ids = self.model.generate(
                inputs['input_ids'],
                max_length=max_length,
                num_beams=4,
                length_penalty=2.0,
                early_stopping=True,
                no_repeat_ngram_size=3
            )
        
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary


def main():
  
    
    # Configuration
    DATA_PATH = "dataset_output/editorial_dataset_training.csv"
    OUTPUT_DIR = "models/finetuned_editorial_t5"
    
    BATCH_SIZE = 4  # Increase to 8 if you have GPU memory
    NUM_EPOCHS = 5
    LEARNING_RATE = 3e-4
    
    # Initialize trainer
    trainer = T5Trainer(model_name='t5-small')
    
    # Prepare data
    train_dataset, val_dataset, test_dataset, test_df = trainer.prepare_data(
        DATA_PATH,
        train_split=0.8,
        val_split=0.1
    )
    
    # Train
    stats = trainer.train(
        train_dataset,
        val_dataset,
        output_dir=OUTPUT_DIR,
        batch_size=BATCH_SIZE,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE
    )
    
    # Test on a sample
    print(f"{'='*80}")
    print("TESTING ON SAMPLE ARTICLE")
    print(f"{'='*80}\n")
    
    sample_article = test_df.iloc[0]['Article']
    sample_summary_gold = test_df.iloc[0]['Summary']
    
    print(f"Article ({len(sample_article.split())} words):")
    print(f"{sample_article[:300]}...\n")
    
    print(f"Gold Summary ({len(sample_summary_gold.split())} words):")
    print(f"{sample_summary_gold}\n")
    
    print(f"Generated Summary (Fine-tuned T5):")
    generated = trainer.generate_summary(sample_article)
    print(f"{generated}\n")
    
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()