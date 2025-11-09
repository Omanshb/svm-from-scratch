import numpy as np
from collections import Counter


def linear_kernel(x1, x2):
    """Linear kernel function."""
    return np.dot(x1, x2)


def polynomial_kernel(x1, x2, degree=3, coef=1):
    """Polynomial kernel function."""
    return (coef + np.dot(x1, x2)) ** degree


def rbf_kernel(x1, x2, gamma=0.1):
    """RBF (Radial Basis Function) kernel."""
    return np.exp(-gamma * np.linalg.norm(x1 - x2) ** 2)


def accuracy_score(y_true, y_pred):
    """Calculate accuracy score."""
    return np.mean(y_true == y_pred)


def precision_score(y_true, y_pred, average='binary'):
    """Calculate precision score."""
    classes = np.unique(y_true)
    
    if average == 'binary' and len(classes) == 2:
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0
    else:
        precisions = []
        for c in classes:
            tp = np.sum((y_true == c) & (y_pred == c))
            fp = np.sum((y_true != c) & (y_pred == c))
            precisions.append(tp / (tp + fp) if (tp + fp) > 0 else 0.0)
        return np.mean(precisions)


def recall_score(y_true, y_pred, average='binary'):
    """Calculate recall score."""
    classes = np.unique(y_true)
    
    if average == 'binary' and len(classes) == 2:
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0
    else:
        recalls = []
        for c in classes:
            tp = np.sum((y_true == c) & (y_pred == c))
            fn = np.sum((y_true == c) & (y_pred != c))
            recalls.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
        return np.mean(recalls)


def f1_score(y_true, y_pred, average='binary'):
    """Calculate F1 score."""
    precision = precision_score(y_true, y_pred, average)
    recall = recall_score(y_true, y_pred, average)
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0


def confusion_matrix(y_true, y_pred):
    """Calculate confusion matrix."""
    classes = np.unique(np.concatenate([y_true, y_pred]))
    n_classes = len(classes)
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    
    for i, true_class in enumerate(classes):
        for j, pred_class in enumerate(classes):
            matrix[i, j] = np.sum((y_true == true_class) & (y_pred == pred_class))
    
    return matrix


class SVMClassifier:
    """
    Support Vector Machine Classifier from scratch.
    Uses Sequential Minimal Optimization (SMO) algorithm with Lagrange multipliers
    to solve the dual optimization problem.
    """
    
    def __init__(self, kernel='linear', C=1.0, learning_rate=0.001, n_iterations=1000,
                 gamma=0.1, degree=3, coef=1, tol=1e-3, random_state=None):
        """
        kernel: kernel type ('linear', 'poly', 'rbf')
        C: regularization parameter (larger C = less regularization)
        learning_rate: learning rate (kept for backward compatibility, not used in SMO)
        n_iterations: maximum number of iterations for SMO
        gamma: kernel coefficient for RBF
        degree: degree for polynomial kernel
        coef: independent term in polynomial kernel
        tol: tolerance for convergence
        random_state: random seed for reproducibility
        """
        self.kernel = kernel
        self.C = C
        self.learning_rate = learning_rate  # Kept for compatibility
        self.n_iterations = n_iterations
        self.gamma = gamma
        self.degree = degree
        self.coef = coef
        self.tol = tol
        self.random_state = random_state
        
        self.w = None
        self.b = None
        self.X_train = None
        self.y_train = None
        self.alpha = None
        self.n_classes_ = None
        self.K = None  # Kernel matrix cache
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def _get_kernel_function(self):
        """Return the appropriate kernel function."""
        if self.kernel == 'linear':
            return lambda x1, x2: linear_kernel(x1, x2)
        elif self.kernel == 'poly':
            return lambda x1, x2: polynomial_kernel(x1, x2, self.degree, self.coef)
        elif self.kernel == 'rbf':
            return lambda x1, x2: rbf_kernel(x1, x2, self.gamma)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")
    
    def _compute_kernel_matrix(self, X1, X2):
        """Compute kernel matrix between X1 and X2."""
        kernel_func = self._get_kernel_function()
        n1, n2 = X1.shape[0], X2.shape[0]
        K = np.zeros((n1, n2))
        
        for i in range(n1):
            for j in range(n2):
                K[i, j] = kernel_func(X1[i], X2[j])
        
        return K
    
    def fit(self, X, y):
        """
        Train the SVM classifier.
        For linear kernel, uses primal form with gradient descent.
        For non-linear kernels, uses dual form.
        """
        X = np.array(X)
        y = np.array(y).flatten()
        
        self.n_classes_ = len(np.unique(y))
        
        if self.n_classes_ == 2:
            y_binary = np.where(y == np.unique(y)[0], -1, 1)
            self._fit_binary(X, y_binary)
        else:
            self._fit_multiclass(X, y)
        
        return self
    
    def _fit_binary(self, X, y):
        """
        Fit binary SVM using Sequential Minimal Optimization (SMO).
        Solves the dual problem:
        maximize: Σα_i - (1/2)ΣΣ α_i α_j y_i y_j K(x_i, x_j)
        subject to: 0 ≤ α_i ≤ C and Σ α_i y_i = 0
        """
        n_samples, n_features = X.shape
        
        # Store training data
        self.X_train = X
        self.y_train = y
        
        # Initialize Lagrange multipliers (alpha)
        self.alpha = np.zeros(n_samples)
        self.b = 0
        
        # Compute and cache kernel matrix
        self.K = self._compute_kernel_matrix(X, X)
        
        # Error cache for efficiency
        self.errors = self._compute_errors()
        
        # SMO algorithm
        num_changed = 0
        examine_all = True
        iteration = 0
        
        while (num_changed > 0 or examine_all) and iteration < self.n_iterations:
            num_changed = 0
            
            if examine_all:
                # Examine all samples
                for i in range(n_samples):
                    num_changed += self._examine_example(i)
            else:
                # Examine non-bound samples (0 < alpha < C)
                non_bound_idx = np.where((self.alpha > self.tol) & 
                                        (self.alpha < self.C - self.tol))[0]
                for i in non_bound_idx:
                    num_changed += self._examine_example(i)
            
            if examine_all:
                examine_all = False
            elif num_changed == 0:
                examine_all = True
            
            iteration += 1
        
        # Compute w for linear kernel (primal form)
        if self.kernel == 'linear':
            self.w = np.sum((self.alpha * self.y_train)[:, np.newaxis] * X, axis=0)
        else:
            self.w = None
    
    def _compute_errors(self):
        """Compute error for all training samples."""
        return self._decision_function_binary(self.X_train) - self.y_train
    
    def _examine_example(self, i2):
        """
        Second choice heuristic for SMO.
        Try to optimize alpha[i2] with another alpha.
        """
        y2 = self.y_train[i2]
        alpha2 = self.alpha[i2]
        E2 = self.errors[i2]
        r2 = E2 * y2
        
        # Check KKT conditions
        # KKT: α=0 => y*f(x)≥1, 0<α<C => y*f(x)=1, α=C => y*f(x)≤1
        if not ((r2 < -self.tol and alpha2 < self.C) or 
                (r2 > self.tol and alpha2 > 0)):
            return 0
        
        # Select i1 using second choice heuristic
        # Choose the one with maximum |E1 - E2|
        non_bound_idx = np.where((self.alpha > self.tol) & 
                                (self.alpha < self.C - self.tol))[0]
        
        i1 = -1
        if len(non_bound_idx) > 1:
            max_step = 0
            for i in non_bound_idx:
                if i == i2:
                    continue
                E1 = self.errors[i]
                step = abs(E1 - E2)
                if step > max_step:
                    max_step = step
                    i1 = i
        
        # If no good i1 found, try random non-bound sample
        if i1 == -1:
            if len(non_bound_idx) > 0:
                i1 = np.random.choice(non_bound_idx)
                if i1 == i2 and len(non_bound_idx) > 1:
                    non_bound_idx = non_bound_idx[non_bound_idx != i2]
                    i1 = np.random.choice(non_bound_idx)
        
        # If still no good i1, try any random sample
        if i1 == -1 or i1 == i2:
            i1 = np.random.randint(0, len(self.alpha))
            while i1 == i2:
                i1 = np.random.randint(0, len(self.alpha))
        
        return self._take_step(i1, i2)
    
    def _take_step(self, i1, i2):
        """
        Optimize alpha[i1] and alpha[i2] jointly.
        This is the core of SMO - analytical solution for 2-variable optimization.
        """
        if i1 == i2:
            return 0
        
        alpha1 = self.alpha[i1]
        alpha2 = self.alpha[i2]
        y1 = self.y_train[i1]
        y2 = self.y_train[i2]
        E1 = self.errors[i1]
        E2 = self.errors[i2]
        s = y1 * y2
        
        # Compute bounds L and H for alpha2
        if y1 != y2:
            L = max(0, alpha2 - alpha1)
            H = min(self.C, self.C + alpha2 - alpha1)
        else:
            L = max(0, alpha1 + alpha2 - self.C)
            H = min(self.C, alpha1 + alpha2)
        
        if L == H:
            return 0
        
        # Compute eta (second derivative of objective function)
        k11 = self.K[i1, i1]
        k12 = self.K[i1, i2]
        k22 = self.K[i2, i2]
        eta = 2 * k12 - k11 - k22
        
        if eta < 0:
            # Normal case: eta is negative
            # Compute new alpha2 (unconstrained)
            a2_new = alpha2 - y2 * (E1 - E2) / eta
            
            # Clip to bounds
            if a2_new >= H:
                a2_new = H
            elif a2_new <= L:
                a2_new = L
        else:
            # Unusual case: must compute objective function at bounds
            # This rarely happens in practice
            a2_new = L if (L + H) / 2 > alpha2 else H
        
        # Check if change is significant
        if abs(a2_new - alpha2) < self.tol * (a2_new + alpha2 + self.tol):
            return 0
        
        # Compute new alpha1
        a1_new = alpha1 + s * (alpha2 - a2_new)
        
        # Update threshold (bias) b
        b1 = self.b - E1 - y1 * (a1_new - alpha1) * k11 - y2 * (a2_new - alpha2) * k12
        b2 = self.b - E2 - y1 * (a1_new - alpha1) * k12 - y2 * (a2_new - alpha2) * k22
        
        if 0 < a1_new < self.C:
            b_new = b1
        elif 0 < a2_new < self.C:
            b_new = b2
        else:
            b_new = (b1 + b2) / 2
        
        # Update error cache
        delta_b = b_new - self.b
        delta1 = y1 * (a1_new - alpha1)
        delta2 = y2 * (a2_new - alpha2)
        
        for i in range(len(self.alpha)):
            if 0 < self.alpha[i] < self.C:
                self.errors[i] += (delta1 * self.K[i1, i] + 
                                  delta2 * self.K[i2, i] + delta_b)
        
        self.errors[i1] = 0  # By definition, error is 0 at support vectors
        self.errors[i2] = 0
        
        # Update model parameters
        self.alpha[i1] = a1_new
        self.alpha[i2] = a2_new
        self.b = b_new
        
        return 1
    
    def _fit_multiclass(self, X, y):
        """Fit multi-class SVM using one-vs-rest strategy."""
        self.classes_ = np.unique(y)
        self.binary_classifiers = []
        
        for class_label in self.classes_:
            y_binary = np.where(y == class_label, 1, -1)
            
            classifier = SVMClassifier(
                kernel=self.kernel,
                C=self.C,
                learning_rate=self.learning_rate,
                n_iterations=self.n_iterations,
                gamma=self.gamma,
                degree=self.degree,
                coef=self.coef,
                tol=self.tol,
                random_state=self.random_state
            )
            classifier._fit_binary(X, y_binary)
            self.binary_classifiers.append(classifier)
    
    def _decision_function_binary(self, X):
        """
        Compute decision function for binary classification.
        f(x) = Σ α_i y_i K(x_i, x) + b
        For linear kernel, this simplifies to w·x + b where w = Σ α_i y_i x_i
        """
        if self.kernel == 'linear' and self.w is not None:
            # Use primal form for efficiency (linear kernel only)
            return np.dot(X, self.w) + self.b
        else:
            # Use dual form (works for all kernels)
            K = self._compute_kernel_matrix(X, self.X_train)
            return np.dot(K, self.alpha * self.y_train) + self.b
    
    def predict(self, X):
        """Predict class labels for samples in X."""
        X = np.array(X)
        
        if self.n_classes_ == 2:
            decision = self._decision_function_binary(X)
            predictions = np.where(decision >= 0, 1, -1)
            
            unique_labels = np.unique(self.y_train)
            if len(unique_labels) == 2:
                predictions = np.where(predictions == 1, unique_labels[1], unique_labels[0])
            
            return predictions
        else:
            n_samples = X.shape[0]
            decision_values = np.zeros((n_samples, len(self.classes_)))
            
            for idx, classifier in enumerate(self.binary_classifiers):
                decision_values[:, idx] = classifier._decision_function_binary(X)
            
            predictions = self.classes_[np.argmax(decision_values, axis=1)]
            return predictions
    
    def decision_function(self, X):
        """Compute decision function values."""
        X = np.array(X)
        
        if self.n_classes_ == 2:
            return self._decision_function_binary(X)
        else:
            n_samples = X.shape[0]
            decision_values = np.zeros((n_samples, len(self.classes_)))
            
            for idx, classifier in enumerate(self.binary_classifiers):
                decision_values[:, idx] = classifier._decision_function_binary(X)
            
            return decision_values
    
    def predict_proba(self, X):
        """
        Predict class probabilities for samples in X.
        Uses Platt scaling approximation.
        """
        X = np.array(X)
        decision_vals = self.decision_function(X)
        
        if self.n_classes_ == 2:
            decision_vals = decision_vals.reshape(-1, 1)
            proba = 1 / (1 + np.exp(-decision_vals))
            return np.hstack([1 - proba, proba])
        else:
            exp_vals = np.exp(decision_vals - np.max(decision_vals, axis=1, keepdims=True))
            return exp_vals / np.sum(exp_vals, axis=1, keepdims=True)
    
    def score(self, X, y):
        """Calculate accuracy score."""
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)
    
    def get_support_vectors(self):
        """
        Get the support vectors.
        Support vectors are points where alpha > 0 (within tolerance).
        """
        if self.n_classes_ == 2:
            if self.alpha is None:
                return None
            support_mask = self.alpha > self.tol
            return self.X_train[support_mask]
        else:
            all_support_vectors = []
            for classifier in self.binary_classifiers:
                if classifier.alpha is not None:
                    support_mask = classifier.alpha > classifier.tol
                    all_support_vectors.append(classifier.X_train[support_mask])
            return all_support_vectors
    
    def get_n_support(self):
        """
        Get the number of support vectors.
        Support vectors are points where alpha > 0 (within tolerance).
        """
        if self.n_classes_ == 2:
            if self.alpha is None:
                return None
            return np.sum(self.alpha > self.tol)
        else:
            n_support = []
            for classifier in self.binary_classifiers:
                if classifier.alpha is not None:
                    n_support.append(np.sum(classifier.alpha > classifier.tol))
            return n_support
    
    def get_params(self):
        """Get model parameters."""
        return {
            'kernel': self.kernel,
            'C': self.C,
            'n_iterations': self.n_iterations,
            'gamma': self.gamma,
            'degree': self.degree,
            'coef': self.coef,
            'tol': self.tol
        }

