import matplotlib.pyplot as plt
import numpy as np
import streamlit as st  # Import Streamlit
def plot_results(results, task):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if task == 'classification':
        # Plot accuracy for classification tasks
        models = list(results.keys())
        accuracies = list(results.values())
        
        ax.bar(models, accuracies, color='skyblue')
        ax.set_xlabel('Models')
        ax.set_ylabel('Accuracy')
        ax.set_title('Model Accuracy Comparison')
        ax.set_xticks(np.arange(len(models)))
        ax.set_xticklabels(models, rotation=45, ha='right')
        for i, v in enumerate(accuracies):
            ax.text(i, v + 0.01, f"{v:.2f}", ha='center', va='bottom')
    
    elif task == 'regression':
        # Plot MSE, RMSE, MAE for regression tasks
        models = list(results.keys())
        mse = [result.get('MSE', 0) for result in results.values()]
        rmse = [result.get('RMSE', 0) for result in results.values()]
        mae = [result.get('MAE', 0) for result in results.values()]
        
        x = np.arange(len(models))
        width = 0.2  # Width of bars
        
        rects1 = ax.bar(x - width, mse, width, label='MSE', color='lightblue')
        rects2 = ax.bar(x, rmse, width, label='RMSE', color='lightgreen')
        rects3 = ax.bar(x + width, mae, width, label='MAE', color='lightcoral')
        
        ax.set_xlabel('Models')
        ax.set_ylabel('Scores')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend()
        
        for rect in rects1 + rects2 + rects3:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    # Ensure the plot is displayed correctly in Streamlit
    st.pyplot(fig)

