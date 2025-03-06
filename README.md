# **Ridge Regression (L2 Regularization) from Scratch using Gradient Descent**

This project implements **Ridge Regression (L2 Regularization)** using **Gradient Descent**, built purely in Python.

---

## **1. Understanding Ridge Regression**

Ridge Regression is an extension of **Linear Regression** that incorporates **L2 regularization** to prevent overfitting by penalizing large coefficient values.

### **Mathematical Formulation**

The cost function for Ridge Regression is:

  $$
  MSE_{ridge} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 + \lambda \sum_{j=1}^{n} b_j^2
  $$

where:

- **yᵢ** is the actual value.
- **ŷᵢ** is the predicted value given by the linear model:

  $$
  \hat{y} = b_0 + b_1X_1 + b_2X_2 + \dots + b_nX_n
  $$

- **λ (lambda)** is the **regularization parameter**, controlling the penalty on large coefficients.
- **b₀, b₁, ..., bₙ** are the model parameters (weights).
- **n** is the total number of data points.

The goal is to **minimize this cost function**, balancing model complexity and accuracy.

---

## **2. Gradient Descent Optimization**

Since Ridge Regression modifies the **Mean Squared Error (MSE)** function by adding an **L2 penalty term**, the gradient update rule changes accordingly.

### **Computing Gradients**

For each coefficient **bⱼ**, the gradient of the Ridge cost function is:

$$
\frac{\partial MSE_{ridge}}{\partial b_j} = -\frac{2}{n} \sum_{i=1}^{n} X_{ij} (y_i - \hat{y}_i) + 2\lambda b_j
$$

where:

- The first term is the standard **Linear Regression** gradient.
- The second term **(2λbⱼ)** is the **L2 regularization** penalty.

### **Updating Coefficients**

Using **Gradient Descent**, the update rule for Ridge Regression becomes:

$$
 b_j = b_j - \alpha \left( -\frac{2}{n} \sum_{i=1}^{n} X_{ij} (y_i - \hat{y}_i) + 2\lambda b_j \right)
$$

where:

- **α (alpha)** is the **learning rate**.
- **λ (lambda)** controls the amount of regularization.

This ensures that the model does not overfit while maintaining a good fit to the data.

---

## **3. Implementation Details**

### **Custom Ridge Regression Class**

We build a **RidgeRegression** class with the following methods:

- **fit(X, y)**: Trains the model using **gradient descent**.
- **predict(X)**: Makes predictions using the trained model.
- **L2 Regularization** is manually applied during the gradient descent step.

---

### **📌 Summary**

✅ Ridge Regression extends Linear Regression by **adding an L2 regularization term**.  
✅ It **prevents overfitting** by **shrinking the magnitude of coefficients**.  
✅ We minimize **Mean Squared Error (MSE) + Regularization Term** to find the best coefficients.  
✅ **Gradient Descent** iteratively updates the coefficients using a modified formula.  
✅ The **regularization parameter (λ)** controls the trade-off between **bias and variance**.  
✅ This implementation is built **from scratch** using only core Python.  

🚀 **Happy Coding!**

