import pandas as pd


'''
Specifying Actions/Arms in the DataFrame
Let’s assume each action or arm corresponds to a different payment strategy, such as:

Payment Gateway A
Payment Gateway B
Payment Gateway C
We can add a column in the DataFrame to represent the chosen action for each transaction. This will allow us to simulate different strategies during training.'''

import pandas as pd

# Sample DataFrame
data = pd.DataFrame({
    'transaction_id': [1, 2, 3, 4, 5],
    'user_id': [101, 102, 103, 104, 105],
    'timestamp': ['2023-07-20 14:23:00', '2023-07-20 09:15:00', '2023-07-21 16:45:00', '2023-07-22 12:30:00', '2023-07-23 19:10:00'],
    'transaction_amount': [100, 200, 150, 300, 250],
    'payment_method': ['credit_card', 'PayPal', 'credit_card', 'PayPal', 'credit_card'],
    'device_type': ['mobile', 'desktop', 'mobile', 'desktop', 'mobile'],
    'location': ['New_York', 'San_Francisco', 'New_York', 'San_Francisco', 'New_York'],
    'previous_transactions': [5, 2, 3, 1, 4],
    'is_successful': [1, 0, 1, 0, 1],
    'action': [0, 1, 0, 2, 1]  # Action represents the payment strategy chosen
})

# Convert timestamp to datetime
data['timestamp'] = pd.to_datetime(data['timestamp'])

# Extract hour and day of the week from timestamp
data['hour'] = data['timestamp'].dt.hour
data['day_of_week'] = data['timestamp'].dt.dayofweek

# One-hot encoding for categorical features
data = pd.get_dummies(data, columns=['payment_method', 'device_type', 'location'])

# Normalize numerical features
from sklearn.preprocessing import StandardScaler

numeric_features = ['transaction_amount', 'previous_transactions', 'hour', 'day_of_week']
scaler = StandardScaler()
data[numeric_features] = scaler.fit_transform(data[numeric_features])

# Define contexts (features) and rewards (target)
contexts = data.drop(columns=['transaction_id', 'user_id', 'timestamp', 'is_successful', 'action']).values
rewards = data['is_successful'].values
actions = data['action'].values



import numpy as np

class EpsilonGreedyLinear:
    def __init__(self, n_arms, n_features, epsilon=0.1):
        self.n_arms = n_arms
        self.n_features = n_features
        self.epsilon = epsilon
        self.arm_rewards = np.zeros(n_arms)
        self.arm_counts = np.zeros(n_arms)
        self.arm_models = [np.zeros(n_features) for _ in range(n_arms)]
        
    def choose_arm(self, context):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_arms)
        else:
            estimated_rewards = [self.arm_models[arm].dot(context) for arm in range(self.n_arms)]
            return np.argmax(estimated_rewards)
    
    def update(self, chosen_arm, reward, context):
        self.arm_counts[chosen_arm] += 1
        alpha = 1.0 / self.arm_counts[chosen_arm]
        self.arm_models[chosen_arm] += alpha * (reward - self.arm_models[chosen_arm].dot(context)) * context

# Define parameters
n_arms = 3  # Example: 3 different payment strategies
n_features = contexts.shape[1]
epsilon = 0.1

# Initialize the algorithm
epsilon_greedy = EpsilonGreedyLinear(n_arms, n_features, epsilon)

# Training phase
for context, reward, action in zip(contexts, rewards, actions):
    chosen_arm = epsilon_greedy.choose_arm(context)
    epsilon_greedy.update(action, reward, context)

# Function to handle a new transaction
def handle_transaction(context):
    chosen_arm = epsilon_greedy.choose_arm(context)
    return chosen_arm

# Example of handling a real-time transaction
new_transaction_context = np.array([0.5, -0.3, 0.8, -1.2, 1, 0, 1, 0, 1, 0, 1, 0])  # Example context vector
chosen_strategy = handle_transaction(new_transaction_context)
print(f"Chosen strategy: {chosen_strategy}")

# After the transaction completes, update the model with the outcome
reward = 1  # For example, if the transaction was successful
epsilon_greedy.update(chosen_strategy, reward, new_transaction_context)