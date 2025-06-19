import boto3
import pandas as pd
import json
import time
from typing import List, Dict, Any, Tuple, Optional
import logging
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import concurrent.futures
from datetime import datetime
import os
import argparse
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BedrockGuardrailTester:
    """
    AWS Data Engineer solution for testing Bedrock Guardrails with performance metrics
    """
    
    def __init__(self, 
                 guardrail_id: str,
                 guardrail_version: str = "DRAFT",
                 aws_region: str = "us-east-1",
                 batch_size: int = 1000):
        """
        Initialize the Bedrock Guardrail Tester
        
        Args:
            guardrail_id: AWS Bedrock Guardrail ID
            guardrail_version: Guardrail version (default: DRAFT)
            aws_region: AWS region
            batch_size: Batch size for processing
        """
        self.guardrail_id = guardrail_id
        self.guardrail_version = guardrail_version
        self.batch_size = batch_size
        self.aws_region = aws_region
        
        # Initialize AWS clients
        self.bedrock_client = boto3.client('bedrock-runtime', region_name=aws_region)
        
        # Performance tracking
        self.performance_metrics = []
        
    def create_guardrail_config(self) -> Dict[str, Any]:
        """
        Create guardrail configuration with high thresholds for all filters
        Note: This returns the configuration structure - actual guardrail creation
        should be done via AWS Console or separate CloudFormation/Terraform
        """
        config = {
            "name": "high-threshold-guardrail",
            "description": "Guardrail with high thresholds for all harmful categories",
            "topicPolicyConfig": {
                "topicsConfig": [
                    {
                        "name": "prompt-attacks",
                        "definition": "Prevent prompt injection and jailbreak attempts",
                        "examples": [],
                        "type": "DENY"
                    }
                ]
            },
            "contentPolicyConfig": {
                "filtersConfig": [
                    {"type": "SEXUAL", "inputStrength": "HIGH", "outputStrength": "HIGH"},
                    {"type": "VIOLENCE", "inputStrength": "HIGH", "outputStrength": "HIGH"},
                    {"type": "HATE", "inputStrength": "HIGH", "outputStrength": "HIGH"},
                    {"type": "INSULTS", "inputStrength": "HIGH", "outputStrength": "HIGH"},
                    {"type": "MISCONDUCT", "inputStrength": "HIGH", "outputStrength": "HIGH"},
                    {"type": "PROMPT_ATTACK", "inputStrength": "HIGH", "outputStrength": "HIGH"}
                ]
            },
            "wordPolicyConfig": {
                "wordsConfig": [],
                "managedWordListsConfig": []
            },
            "sensitiveInformationPolicyConfig": {
                "piiEntitiesConfig": [],
                "regexesConfig": []
            },
            "blockedInputMessaging": "Your message was blocked due to policy violations.",
            "blockedOutputsMessaging": "The response was blocked due to policy violations."
        }
        return config
    
    def load_dataset(self, file_path: str) -> pd.DataFrame:
        """
        Load dataset from various formats (CSV, Parquet, JSONL)
        
        Args:
            file_path: Path to the dataset file
            
        Returns:
            DataFrame with loaded data
        """
        file_extension = Path(file_path).suffix.lower()
        
        try:
            if file_extension == '.csv':
                df = pd.read_csv(file_path)
            elif file_extension == '.parquet':
                df = pd.read_parquet(file_path)
            elif file_extension == '.jsonl':
                df = pd.read_json(file_path, lines=True)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
            
            logger.info(f"Loaded {len(df)} records from {file_path}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading dataset {file_path}: {str(e)}")
            raise
    
    def auto_discover_columns(self, df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
        """
        Auto-discover text column and ground truth column from the DataFrame
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (text_column, ground_truth_column)
        """
        text_column = None
        ground_truth_column = None
        
        # Common text column names (case-insensitive)
        text_column_candidates = [
            'text', 'content', 'prompt', 'input', 'message', 'query', 
            'question', 'statement', 'sentence', 'paragraph', 'data',
            'text_content', 'input_text', 'prompt_text', 'user_input'
        ]
        
        # Common ground truth column names (case-insensitive)
        ground_truth_candidates = [
            'label', 'ground_truth', 'is_harmful', 'is_toxic', 'is_blocked',
            'harmful', 'toxic', 'blocked', 'flagged', 'violation',
            'target', 'class', 'category', 'classification', 'gt',
            'ground_truth_label', 'true_label', 'expected', 'annotation'
        ]
        
        # Convert column names to lowercase for comparison
        df_columns_lower = [col.lower() for col in df.columns]
        column_mapping = {col.lower(): col for col in df.columns}
        
        # Find text column
        for candidate in text_column_candidates:
            if candidate.lower() in df_columns_lower:
                text_column = column_mapping[candidate.lower()]
                logger.info(f"Auto-discovered text column: '{text_column}'")
                break
        
        # If no exact match, look for columns containing text-like keywords
        if not text_column:
            for col_lower, col_original in column_mapping.items():
                if any(keyword in col_lower for keyword in ['text', 'content', 'prompt', 'input']):
                    text_column = col_original
                    logger.info(f"Auto-discovered text column (partial match): '{text_column}'")
                    break
        
        # If still no text column, use the first string column with substantial content
        if not text_column:
            for col in df.columns:
                if df[col].dtype == 'object':
                    # Check if column contains substantial text (average length > 10 chars)
                    sample_data = df[col].dropna().head(100)
                    if len(sample_data) > 0:
                        avg_length = sample_data.astype(str).str.len().mean()
                        if avg_length > 10:
                            text_column = col
                            logger.info(f"Auto-discovered text column (by content analysis): '{text_column}'")
                            break
        
        # Find ground truth column
        for candidate in ground_truth_candidates:
            if candidate.lower() in df_columns_lower:
                ground_truth_column = column_mapping[candidate.lower()]
                logger.info(f"Auto-discovered ground truth column: '{ground_truth_column}'")
                break
        
        # If no exact match, look for boolean or binary columns
        if not ground_truth_column:
            for col in df.columns:
                if col.lower() != text_column.lower() if text_column else True:
                    unique_values = df[col].dropna().unique()
                    # Check if column is binary (boolean, 0/1, true/false, etc.)
                    if len(unique_values) == 2:
                        # Check if values look like boolean/binary labels
                        unique_str = [str(v).lower() for v in unique_values]
                        if (set(unique_str).issubset({'0', '1', 'true', 'false', 'yes', 'no', 
                                                     'harmful', 'safe', 'toxic', 'clean'})):
                            ground_truth_column = col
                            logger.info(f"Auto-discovered ground truth column (binary analysis): '{ground_truth_column}'")
                            break
        
        # Print column discovery summary
        logger.info(f"Column discovery summary:")
        logger.info(f"  Available columns: {list(df.columns)}")
        logger.info(f"  Text column: {text_column}")
        logger.info(f"  Ground truth column: {ground_truth_column}")
        
        if not text_column:
            logger.error("Could not auto-discover text column. Please check your dataset.")
            logger.info("Available columns and their sample data:")
            for col in df.columns[:5]:  # Show first 5 columns
                sample = df[col].dropna().head(3).tolist()
                logger.info(f"  {col}: {sample}")
        
        return text_column, ground_truth_column
    
    def test_single_prompt(self, text: str) -> Dict[str, Any]:
        """
        Test a single prompt against Bedrock Guardrail
        
        Args:
            text: Input text to test
            
        Returns:
            Dictionary with guardrail response and metadata
        """
        start_time = time.time()
        
        try:
            response = self.bedrock_client.apply_guardrail(
                guardrailIdentifier=self.guardrail_id,
                guardrailVersion=self.guardrail_version,
                source='INPUT',
                content=[{
                    'text': {
                        'text': text
                    }
                }]
            )
            
            end_time = time.time()
            latency_ms = (end_time - start_time) * 1000
            
            # Parse response
            action = response.get('action', 'NONE')
            is_blocked = action == 'GUARDRAIL_INTERVENED'
            
            # Extract assessment details
            assessments = response.get('assessments', [])
            
            result = {
                'text': text,
                'action': action,
                'is_blocked': is_blocked,
                'latency_ms': latency_ms,
                'assessments': assessments,
                'confidence': self._extract_confidence(assessments),
                'strength': self._extract_strength(assessments),
                'detected_categories': self._extract_categories(assessments),
                'timestamp': datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error testing prompt: {str(e)}")
            return {
                'text': text,
                'action': 'ERROR',
                'is_blocked': False,
                'latency_ms': 0,
                'assessments': [],
                'confidence': 0,
                'strength': 'NONE',
                'detected_categories': [],
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _extract_confidence(self, assessments: List[Dict]) -> float:
        """Extract maximum confidence from assessments"""
        confidences = []
        for assessment in assessments:
            if 'topicPolicy' in assessment:
                for topic in assessment['topicPolicy'].get('topics', []):
                    confidences.append(topic.get('confidence', 0))
            if 'contentPolicy' in assessment:
                for filter_item in assessment['contentPolicy'].get('filters', []):
                    confidences.append(filter_item.get('confidence', 0))
        
        return max(confidences) if confidences else 0.0
    
    def _extract_strength(self, assessments: List[Dict]) -> str:
        """Extract maximum strength from assessments"""
        strengths = []
        for assessment in assessments:
            if 'contentPolicy' in assessment:
                for filter_item in assessment['contentPolicy'].get('filters', []):
                    strengths.append(filter_item.get('strength', 'NONE'))
        
        strength_order = {'NONE': 0, 'LOW': 1, 'MEDIUM': 2, 'HIGH': 3}
        if strengths:
            return max(strengths, key=lambda x: strength_order.get(x, 0))
        return 'NONE'
    
    def _extract_categories(self, assessments: List[Dict]) -> List[str]:
        """Extract detected categories from assessments"""
        categories = []
        for assessment in assessments:
            if 'contentPolicy' in assessment:
                for filter_item in assessment['contentPolicy'].get('filters', []):
                    if filter_item.get('action') == 'BLOCKED':
                        categories.append(filter_item.get('type', 'UNKNOWN'))
        
        return categories
    
    def process_batch(self, batch_df: pd.DataFrame, text_column: str, 
                     ground_truth_column: str = None) -> List[Dict[str, Any]]:
        """
        Process a batch of prompts
        
        Args:
            batch_df: DataFrame containing the batch
            text_column: Column name containing text to test
            ground_truth_column: Column name containing ground truth labels
            
        Returns:
            List of results
        """
        results = []
        
        for idx, row in batch_df.iterrows():
            text = str(row[text_column])
            ground_truth = row.get(ground_truth_column) if ground_truth_column else None
            
            result = self.test_single_prompt(text)
            
            if ground_truth is not None:
                result['ground_truth'] = ground_truth
                result['ground_truth_blocked'] = bool(ground_truth)
            
            results.append(result)
            
            # Log progress
            if (idx + 1) % 100 == 0:
                logger.info(f"Processed {idx + 1} prompts in current batch")
        
        return results
    
    def calculate_performance_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate performance metrics comparing ground truth vs guardrail results
        
        Args:
            results: List of test results
            
        Returns:
            Dictionary with performance metrics
        """
        # Filter results that have ground truth
        results_with_gt = [r for r in results if 'ground_truth' in r and r.get('action') != 'ERROR']
        
        if not results_with_gt:
            logger.warning("No ground truth data available for performance calculation")
            return {}
        
        # Extract predictions and ground truth
        y_true = [r['ground_truth_blocked'] for r in results_with_gt]
        y_pred = [r['is_blocked'] for r in results_with_gt]
        
        # Calculate latencies
        latencies = [r['latency_ms'] for r in results_with_gt if r['latency_ms'] > 0]
        avg_latency = np.mean(latencies) if latencies else 0
        
        # Calculate confusion matrix components
        tp = sum(1 for t, p in zip(y_true, y_pred) if t and p)  # True Positive
        tn = sum(1 for t, p in zip(y_true, y_pred) if not t and not p)  # True Negative
        fp = sum(1 for t, p in zip(y_true, y_pred) if not t and p)  # False Positive
        fn = sum(1 for t, p in zip(y_true, y_pred) if t and not p)  # False Negative
        
        total = len(y_true)
        
        # Calculate metrics
        metrics = {
            'accuracy_percent': accuracy_score(y_true, y_pred) * 100,
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
            'false_negative_rate': fn / (fn + tp) if (fn + tp) > 0 else 0,
            'true_positive_rate': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'true_negative_rate': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'avg_latency_ms': avg_latency,
            'total_samples': total,
            'true_positives': tp,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn
        }
        
        return metrics
    
    def run_comprehensive_test(self, 
                              dataset_path: str, 
                              text_column: str = None,
                              ground_truth_column: str = None,
                              output_path: str = None) -> Dict[str, Any]:
        """
        Run comprehensive testing on the dataset
        
        Args:
            dataset_path: Path to the dataset file
            text_column: Column name containing text to test (auto-discovered if None)
            ground_truth_column: Column name containing ground truth labels (auto-discovered if None)
            output_path: Path for output CSV file (auto-generated if None)
            
        Returns:
            Summary of results and performance metrics
        """
        logger.info(f"Starting comprehensive test on {dataset_path}")
        
        # Auto-generate output path if not provided
        if output_path is None:
            dataset_name = Path(dataset_path).stem
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"guardrail_test_results_{dataset_name}_{timestamp}.csv"
        
        # Load dataset
        df = self.load_dataset(dataset_path)
        total_records = len(df)
        
        # Auto-discover columns if not provided
        if text_column is None or ground_truth_column is None:
            discovered_text_col, discovered_gt_col = self.auto_discover_columns(df)
            
            if text_column is None:
                text_column = discovered_text_col
            if ground_truth_column is None:
                ground_truth_column = discovered_gt_col
        
        # Validate text column
        if text_column is None or text_column not in df.columns:
            raise ValueError(f"Text column '{text_column}' not found in dataset. Available columns: {list(df.columns)}")
        
        # Validate ground truth column (optional)
        if ground_truth_column and ground_truth_column not in df.columns:
            logger.warning(f"Ground truth column '{ground_truth_column}' not found. Performance metrics will not be calculated.")
            ground_truth_column = None
        
        logger.info(f"Using text column: '{text_column}'")
        logger.info(f"Using ground truth column: '{ground_truth_column}'" if ground_truth_column else "No ground truth column - performance metrics will not be calculated")
        logger.info(f"Processing {total_records} records in batches of {self.batch_size}")
        
        all_results = []
        
        # Process in batches
        for i in range(0, total_records, self.batch_size):
            batch_start = i
            batch_end = min(i + self.batch_size, total_records)
            
            logger.info(f"Processing batch {batch_start + 1}-{batch_end} of {total_records}")
            
            batch_df = df.iloc[batch_start:batch_end].copy()
            batch_results = self.process_batch(batch_df, text_column, ground_truth_column)
            
            all_results.extend(batch_results)
            
            # Save intermediate results every 5 batches
            if (i // self.batch_size + 1) % 5 == 0:
                temp_df = pd.DataFrame(all_results)
                temp_output = output_path.replace('.csv', f'_temp_batch_{i//self.batch_size + 1}.csv')
                temp_df.to_csv(temp_output, index=False)
                logger.info(f"Saved intermediate results to {temp_output}")
        
        # Calculate performance metrics
        performance_metrics = self.calculate_performance_metrics(all_results)
        
        # Create results DataFrame
        results_df = pd.DataFrame(all_results)
        
        # Add performance summary to the DataFrame
        if performance_metrics:
            summary_row = {
                'text': 'PERFORMANCE_SUMMARY',
                'action': 'SUMMARY',
                'is_blocked': None,
                'latency_ms': performance_metrics.get('avg_latency_ms', 0),
                'assessments': json.dumps(performance_metrics),
                'confidence': None,
                'strength': None,
                'detected_categories': [],
                'timestamp': datetime.now().isoformat()
            }
            
            if ground_truth_column:
                summary_row['ground_truth'] = None
                summary_row['ground_truth_blocked'] = None
            
            results_df = pd.concat([results_df, pd.DataFrame([summary_row])], ignore_index=True)
        
        # Save results
        results_df.to_csv(output_path, index=False)
        logger.info(f"Results saved to {output_path}")
        
        # Print performance summary
        if performance_metrics:
            logger.info("Performance Metrics Summary:")
            logger.info(f"Accuracy: {performance_metrics['accuracy_percent']:.2f}%")
            logger.info(f"Precision: {performance_metrics['precision']:.4f}")
            logger.info(f"Recall: {performance_metrics['recall']:.4f}")
            logger.info(f"F1 Score: {performance_metrics['f1_score']:.4f}")
            logger.info(f"False Positive Rate: {performance_metrics['false_positive_rate']:.4f}")
            logger.info(f"False Negative Rate: {performance_metrics['false_negative_rate']:.4f}")
            logger.info(f"Average Latency: {performance_metrics['avg_latency_ms']:.2f} ms")
        
        return {
            'total_processed': len(all_results),
            'performance_metrics': performance_metrics,
            'output_file': output_path,
            'text_column_used': text_column,
            'ground_truth_column_used': ground_truth_column,
            'results_summary': {
                'blocked_count': sum(1 for r in all_results if r['is_blocked']),
                'error_count': sum(1 for r in all_results if r['action'] == 'ERROR'),
                'avg_confidence': np.mean([r['confidence'] for r in all_results if r['confidence'] > 0])
            }
        }


def parse_arguments():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(
        description="AWS Bedrock Guardrail Testing Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python guardrail_tester.py --dataset data.csv --guardrail-id gr-123456789
    python guardrail_tester.py --dataset data.parquet --guardrail-id gr-123456789 --batch-size 500
    python guardrail_tester.py --dataset data.jsonl --guardrail-id gr-123456789 --text-column "prompt" --gt-column "harmful"
    python guardrail_tester.py --dataset data.csv --guardrail-id gr-123456789 --output results.csv --region us-west-2
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--dataset', '-d',
        required=True,
        help='Path to the dataset file (CSV, Parquet, or JSONL)'
    )
    
    parser.add_argument(
        '--guardrail-id', '-g',
        required=True,
        help='AWS Bedrock Guardrail ID'
    )
    
    # Optional arguments
    parser.add_argument(
        '--guardrail-version', '-v',
        default='DRAFT',
        help='Guardrail version (default: DRAFT)'
    )
    
    parser.add_argument(
        '--region', '-r',
        default='us-east-1',
        help='AWS region (default: us-east-1)'
    )
    
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=1000,
        help='Batch size for processing (default: 1000)'
    )
    
    parser.add_argument(
        '--text-column', '-t',
        help='Column name containing text to test (auto-discovered if not specified)'
    )
    
    parser.add_argument(
        '--gt-column', '--ground-truth-column',
        help='Column name containing ground truth labels (auto-discovered if not specified)'
    )
    
    parser.add_argument(
        '--output', '-o',
        help='Output CSV file path (auto-generated if not specified)'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    
    return parser.parse_args()


def main():
    """
    Main function with command line argument parsing
    """
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Configure logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Validate dataset file exists
    if not os.path.exists(args.dataset):
        logger.error(f"Dataset file not found: {args.dataset}")
        sys.exit(1)
    
    # Validate file format
    supported_extensions = ['.csv', '.parquet', '.jsonl']
    file_extension = Path(args.dataset).suffix.lower()
    if file_extension not in supported_extensions:
        logger.error(f"Unsupported file format: {file_extension}. Supported formats: {supported_extensions}")
        sys.exit(1)
    
    try:
        logger.info("=== AWS Bedrock Guardrail Testing Script ===")
        logger.info(f"Dataset: {args.dataset}")
        logger.info(f"Guardrail ID: {args.guardrail_id}")
        logger.info(f"Guardrail Version: {args.guardrail_version}")
        logger.info(f"AWS Region: {args.region}")
        logger.info(f"Batch Size: {args.batch_size}")
        
        # Initialize tester
        tester = BedrockGuardrailTester(
            guardrail_id=args.guardrail_id,
            guardrail_version=args.guardrail_version,
            aws_region=args.region,
            batch_size=args.batch_size
        )
        
        # Print guardrail configuration example
        if args.log_level == 'DEBUG':
            config = tester.create_guardrail_config()
            logger.debug("Example Guardrail Configuration:")
            logger.debug(json.dumps(config, indent=2))
        
        # Run comprehensive test
        results = tester.run_comprehensive_test(
            dataset_path=args.dataset,
            text_column=args.text_column,
            ground_truth_column=args.gt_column,
            output_path=args.output
        )
        
        logger.info("=== Testing Completed Successfully! ===")
        logger.info(f"Total records processed: {results['total_processed']}")
        logger.info(f"Output file: {results['output_file']}")
        logger.info(f"Text column used: {results['text_column_used']}")
        logger.info(f"Ground truth column used: {results['ground_truth_column_used']}")
        logger.info(f"Blocked prompts: {results['results_summary']['blocked_count']}")
        logger.info(f"Errors: {results['results_summary']['error_count']}")
        
        if results['performance_metrics']:
            logger.info("=== Final Performance Summary ===")
            metrics = results['performance_metrics']
            logger.info(f"Accuracy: {metrics['accuracy_percent']:.2f}%")
            logger.info(f"Precision: {metrics['precision']:.4f}")
            logger.info(f"Recall: {metrics['recall']:.4f}")
            logger.info(f"F1 Score: {metrics['f1_score']:.4f}")
            logger.info(f"Average Latency: {metrics['avg_latency_ms']:.2f} ms")
        
    except KeyboardInterrupt:
        logger.info("Script interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        if args.log_level == 'DEBUG':
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()