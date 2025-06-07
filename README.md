# Double Q-Learning Chess Bot

This project implements a simplified chess-playing agent using the **Double Q-Learning** algorithm. It was developed as part of a university coursework project on Reinforcement Learning in Games.

## 🎯 Project Objective

The aim is to train an agent that learns to make decisions in a chess-like environment using two Q-tables to reduce overestimation bias in action-value estimates.

## 🧠 Algorithm Used

- **Double Q-Learning**: A variant of Q-learning that maintains two separate value functions (Q1 and Q2) to improve learning stability and reduce overestimation.

## 🗂️ Files

- `doubleqlearning.py` – Main script that runs the training loop, updates Q-tables, and tests performance.
- `games.csv` – Dataset used to track or simulate game outcomes and agent decisions.

## ⚙️ How to Run

```bash
python doubleqlearning.py
```

Make sure you have Python 3.x installed and the required libraries listed below.

## 📦 Dependencies

- `numpy`
- `pandas`

You can install them using:

```bash
pip install numpy pandas
```

## 👨‍💻 Author

This project was developed by Sueda Berra Aygun as part of the COMP1827 module at University of Greenwich.
