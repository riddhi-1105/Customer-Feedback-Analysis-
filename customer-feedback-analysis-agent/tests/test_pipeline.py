"""Quick end-to-end pipeline test"""
import pandas as pd
import sys
sys.path.insert(0, '.')
from agents.customer_feedback_agent import CustomerFeedbackAgent

df = pd.read_csv('data/sample_feedback.csv')
print(f'Loaded {len(df)} records')

agent = CustomerFeedbackAgent()
results = agent.analyze_dataset(df, 'feedback', 'date', 'rating')

analyzed_df = results['df']
print(f'Analyzed: {len(analyzed_df)} records')
print('Sentiment col:', 'sentiment' in analyzed_df.columns)
print('Emotion col:', 'emotion' in analyzed_df.columns)
print('Topic col:', 'topic' in analyzed_df.columns)
print('Issue col:', 'detected_issue' in analyzed_df.columns)
print('Priority col:', 'priority' in analyzed_df.columns)
print('Recurring issues:', len(results['recurring_issues']))
print('Insights:', len(results['insights']))
print('Recommendations:', len(results['recommendations']))
print('Sentiment distribution:', analyzed_df['sentiment'].value_counts().to_dict())
print('Priority distribution:', results['priority_distribution'])
print()
print('Activity log:')
for step in results['activity_log']:
    print(f"  Step {step['step']}: [{step['status']}] {step['description'][:80]}")
print()
print('SUCCESS: Full pipeline works!')
