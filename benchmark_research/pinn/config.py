import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'historical_data', 'traffic_history_action.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results', 'pinn')

# Model Parameters
INPUT_DIM = 10
# Wait, episode and intersection_id shouldn't be used as continuous features directly. 
# We should use: traffic_density, queue_length, throughput, avg_speed, fuel_consumption, current_phase, phase_duration
# Actually, the user said inputs: episode, simulation time, step number, intersection ID, traffic density, queue length, throughput, average speed, fuel consumption, current phase, and phase duration.
# We will encode intersection_id, and pass others.
HIDDEN_DIM = 128
NUM_LAYERS = 5
OUTPUT_DIM = 5  # density(t+1), queue(t+1), throughput(t+1), speed(t+1), fuel(t+1)
# Wait, travel_time is NOT in our dataset because we couldn't easily get it per step!
# Ah, I forgot about travel time. Let me check if I added travel_time to generate_pinn_dataset.py.
# I did not add travel_time because it's only logged when vehicles complete their journey.
# Let me look closely at the generated CSV...

# Hyperparameters
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
EPOCHS = 200
PHYSICS_LAMBDA = 0.5  # Weight for the Physics Loss
PATIENCE = 10  # For Early Stopping

# Data Splitting
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
