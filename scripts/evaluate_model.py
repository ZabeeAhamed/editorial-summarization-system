import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime

import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from rouge_score import rouge_scorer
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class ModelEvaluator:
    
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("="*80)
        print("📊 EDITORIAL SUMMARIZATION - MODEL EVALUATION")
        print("="*80)
        print(f"Author: ZabeeAhmed")
        print(f"Date: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
        print(f"Device: {self.device}")
        print("="*80 + "\n")
        
        # ROUGE scorer
        self.scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'],
            use_stemmer=True
        )
    
    def load_models(self, pretrained_name='t5-small', finetuned_path=None):
        #Loading both pretrained and fine-tuned models
        
        print("Loading models...\n")
        
        # Pretrained model
        print("  Loading pretrained T5-small...")
        self.pretrained_tokenizer = T5Tokenizer.from_pretrained(pretrained_name, legacy=False)
        self.pretrained_model = T5ForConditionalGeneration.from_pretrained(pretrained_name)
        self.pretrained_model.to(self.device)
        self.pretrained_model.eval()
        print(" Pretrained model loaded\n")
        
        # Fine-tuned model
        if finetuned_path:
            print(f"  Loading fine-tuned model from: {finetuned_path}")
            self.finetuned_tokenizer = T5Tokenizer.from_pretrained(finetuned_path, legacy=False)
            self.finetuned_model = T5ForConditionalGeneration.from_pretrained(finetuned_path)
            self.finetuned_model.to(self.device)
            self.finetuned_model.eval()
            print(" Fine-tuned model loaded\n")
        else:
            self.finetuned_model = None
            self.finetuned_tokenizer = None
    
    def generate_summary(self, model, tokenizer, text, max_length=150):

        input_text = f"summarize: {text}"        
        inputs = tokenizer(
            input_text,
            max_length=512,
            truncation=True,
            return_tensors='pt'
        ).to(self.device)
        
        with torch.no_grad():
            summary_ids = model.generate(
                inputs['input_ids'],
                max_length=max_length,
                num_beams=4,
                length_penalty=2.0,
                early_stopping=True,
                no_repeat_ngram_size=3
            )
        
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
    
    def calculate_rouge(self, prediction, reference):
        scores = self.scorer.score(reference, prediction)
        return {
            'rouge1_f': scores['rouge1'].fmeasure,
            'rouge1_p': scores['rouge1'].precision,
            'rouge1_r': scores['rouge1'].recall,
            'rouge2_f': scores['rouge2'].fmeasure,
            'rouge2_p': scores['rouge2'].precision,
            'rouge2_r': scores['rouge2'].recall,
            'rougeL_f': scores['rougeL'].fmeasure,
            'rougeL_p': scores['rougeL'].precision,
            'rougeL_r': scores['rougeL'].recall,
        }
    
    def evaluate_on_dataset(self, test_df):

        print("Evaluating models on test set...\n")
        results = []
        
        for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Evaluating"):
            article = row['Article']
            reference_summary = row['Summary']
            
            # Generate with pretrained
            pretrained_summary = self.generate_summary(
                self.pretrained_model,
                self.pretrained_tokenizer,
                article
            )
            
            # Generate with fine-tuned
            if self.finetuned_model:
                finetuned_summary = self.generate_summary(
                    self.finetuned_model,
                    self.finetuned_tokenizer,
                    article
                )
            else:
                finetuned_summary = ""
            
            # Calculate ROUGE scores
            pretrained_rouge = self.calculate_rouge(pretrained_summary, reference_summary)
            
            if finetuned_summary:
                finetuned_rouge = self.calculate_rouge(finetuned_summary, reference_summary)
            else:
                finetuned_rouge = {}
            
            results.append({
                'article': article,
                'reference': reference_summary,
                'pretrained_summary': pretrained_summary,
                'finetuned_summary': finetuned_summary,
                'pretrained_rouge': pretrained_rouge,
                'finetuned_rouge': finetuned_rouge
            })
        
        return results
    
    def aggregate_scores(self, results):
        """Aggregate ROUGE scores"""
        
        pretrained_scores = {
            'rouge1_f': [], 'rouge2_f': [], 'rougeL_f': []
        }
        
        finetuned_scores = {
            'rouge1_f': [], 'rouge2_f': [], 'rougeL_f': []
        }
        
        for result in results:
            pretrained_scores['rouge1_f'].append(result['pretrained_rouge']['rouge1_f'])
            pretrained_scores['rouge2_f'].append(result['pretrained_rouge']['rouge2_f'])
            pretrained_scores['rougeL_f'].append(result['pretrained_rouge']['rougeL_f'])
            
            if result['finetuned_rouge']:
                finetuned_scores['rouge1_f'].append(result['finetuned_rouge']['rouge1_f'])
                finetuned_scores['rouge2_f'].append(result['finetuned_rouge']['rouge2_f'])
                finetuned_scores['rougeL_f'].append(result['finetuned_rouge']['rougeL_f'])
        
        # Calculate means
        pretrained_avg = {k: np.mean(v) for k, v in pretrained_scores.items()}
        finetuned_avg = {k: np.mean(v) for k, v in finetuned_scores.items()} if finetuned_scores['rouge1_f'] else {}
        
        return pretrained_avg, finetuned_avg
    
    def print_results(self, pretrained_avg, finetuned_avg):
        """Print comparison table"""
        
        print("\n" + "="*80)
        print(" EVALUATION RESULTS - ROUGE SCORES")
        print("="*80)
        
        print("\n PRETRAINED T5-SMALL (Baseline)")
        print("-"*80)
        print(f"  ROUGE-1 F1: {pretrained_avg['rouge1_f']:.4f}")
        print(f"  ROUGE-2 F1: {pretrained_avg['rouge2_f']:.4f}")
        print(f"  ROUGE-L F1: {pretrained_avg['rougeL_f']:.4f}")
        
        if finetuned_avg:
            print("\n FINE-TUNED T5 (Editorial-adapted)")
            print("-"*80)
            print(f"  ROUGE-1 F1: {finetuned_avg['rouge1_f']:.4f}")
            print(f"  ROUGE-2 F1: {finetuned_avg['rouge2_f']:.4f}")
            print(f"  ROUGE-L F1: {finetuned_avg['rougeL_f']:.4f}")
            
            print("\n IMPROVEMENT (Fine-tuned vs Pretrained)")
            print("-"*80)
            
            improvements = {
                'ROUGE-1': ((finetuned_avg['rouge1_f'] - pretrained_avg['rouge1_f']) / pretrained_avg['rouge1_f']) * 100,
                'ROUGE-2': ((finetuned_avg['rouge2_f'] - pretrained_avg['rouge2_f']) / pretrained_avg['rouge2_f']) * 100,
                'ROUGE-L': ((finetuned_avg['rougeL_f'] - pretrained_avg['rougeL_f']) / pretrained_avg['rougeL_f']) * 100,
            }
            
            for metric, improvement in improvements.items():
                symbol = "📈" if improvement > 0 else "📉"
                print(f"  {symbol} {metric}: {improvement:+.2f}%")
        
        print("\n" + "="*80)
    
    def save_results(self, results, pretrained_avg, finetuned_avg, output_dir):
        
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save aggregate scores
        scores_path = output_dir / 'evaluation_scores.json'
        with open(scores_path, 'w') as f:
            json.dump({
                'pretrained': pretrained_avg,
                'finetuned': finetuned_avg,
                'evaluation_date': datetime.utcnow().isoformat(),
                'author': 'ZabeeAhmed'
            }, f, indent=2)
        
        print(f"\n Results saved:")
        print(f" {scores_path}")
        
        return scores_path


def main():

    
    # Configuration
    DATA_PATH = "dataset_output/editorial_dataset_training_cleaned.csv"
    FINETUNED_MODEL_PATH = "models/finetuned_editorial_t5"
    OUTPUT_DIR = "results"
    
    # Load data
    print(" Loading test data...")
    df = pd.read_csv(DATA_PATH)
    
    # Use last 10% as test set
    n = len(df)
    train_size = int(n * 0.8)
    val_size = int(n * 0.1)
    test_df = df[train_size + val_size:].reset_index(drop=True)
    
    print(f"   Test samples: {len(test_df)}\n")
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Load models
    evaluator.load_models(
        pretrained_name='t5-small',
        finetuned_path=FINETUNED_MODEL_PATH
    )
    
    # Evaluate
    results = evaluator.evaluate_on_dataset(test_df)
    
    # Aggregate scores
    pretrained_avg, finetuned_avg = evaluator.aggregate_scores(results)
    
    # Print results
    evaluator.print_results(pretrained_avg, finetuned_avg)
    
    # Save results
    evaluator.save_results(results, pretrained_avg, finetuned_avg, OUTPUT_DIR)
    
    print("\n Evaluation complete!\n")


if __name__ == "__main__":
    main()