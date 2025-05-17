# RL Snake AI - Teach AI to Play Snake!

This project demonstrates how to teach an AI agent to play the classic Snake game using **Reinforcement Learning**. The implementation is built from scratch using **PyTorch** and **Pygame**.

## Features
- **Custom Snake Game Environment**: Built using Pygame.
- **Deep Q-Learning Agent**: Implements a neural network to predict moves and improve performance over time.
- **Visualization**: Real-time game display and training progress plotting.

## Requirements
To run this project, you need the following dependencies:
- Python 3.8+
- PyTorch == 2.7.0
- Pygame == 2.6.1
- NumPy == 2.2.5
- Matplotlib == 3.10.3
- Other dependencies listed in `requirements.txt`

Install the required packages using:
```bash
pip install -r requirements.txt
```

## How to Generate `requirements.txt`
If you need to update the `requirements.txt` file, run the following command in your virtual environment:
```bash
pip freeze > requirements.txt
```

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/rl-snake-ai.git
   cd rl-snake-ai
   ```
2. Activate your virtual environment:
   ```bash
   .\venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the training script:
   ```bash
   python train.py
   ```

## File Structure
- `agent.py`: Contains the implementation of the RL agent.
- `game.py`: Implements the Snake game environment.
- `model.py`: Defines the neural network architecture.
- `helper.py`: Utility functions for plotting and debugging.
- `requirements.txt`: Lists all dependencies for the project.

## Acknowledgments
This project is inspired by the tutorial series on teaching AI to play Snake using Reinforcement Learning. Special thanks to the creators of PyTorch and Pygame for their amazing libraries.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.
