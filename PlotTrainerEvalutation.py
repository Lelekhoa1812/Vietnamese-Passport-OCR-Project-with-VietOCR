import matplotlib.pyplot as plt

plt.plot(trainer.history['train_loss'], label='Train Loss')
plt.plot(trainer.history['valid_loss'], label='Validation Loss')
plt.legend()
plt.show()