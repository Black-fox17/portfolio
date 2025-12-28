export interface BlogPost {
  id: string;
  title: string;
  date: string;
  readTime: string;
  category: string;
  excerpt: string;
  content: string;
}

export const blogPosts: BlogPost[] = [
  {
    id: "1",
    title: "Understanding Backpropagation: The Mathematics Behind Neural Networks",
    date: "2024-12-15",
    readTime: "8 min read",
    category: "Deep Learning",
    excerpt: "A deep dive into the mathematical foundations of backpropagation, exploring the chain rule and gradient descent that power modern neural networks.",
    content: `
# Understanding Backpropagation: The Mathematics Behind Neural Networks

Backpropagation is the cornerstone algorithm that enables neural networks to learn. In this post, we'll explore the mathematical foundations that make deep learning possible.

## The Chain Rule Foundation

At its core, backpropagation is an elegant application of the chain rule from calculus. For a composite function $f(g(x))$, the derivative is:

$$\\frac{df}{dx} = \\frac{df}{dg} \\cdot \\frac{dg}{dx}$$

In neural networks, we have layers of transformations, making this particularly powerful.

## Forward Pass

Consider a simple neural network with one hidden layer:

$$z^{[1]} = W^{[1]}x + b^{[1]}$$
$$a^{[1]} = \\sigma(z^{[1]})$$
$$z^{[2]} = W^{[2]}a^{[1]} + b^{[2]}$$
$$\\hat{y} = \\sigma(z^{[2]})$$

Where $\\sigma$ is the activation function (e.g., sigmoid or ReLU).

## Loss Function

We typically use cross-entropy loss for classification:

$$L(\\hat{y}, y) = -\\sum_{i} y_i \\log(\\hat{y}_i)$$

For regression, mean squared error is common:

$$L(\\hat{y}, y) = \\frac{1}{2}(\\hat{y} - y)^2$$

## Backward Pass

The magic happens when we compute gradients. For the output layer:

$$\\frac{\\partial L}{\\partial W^{[2]}} = \\frac{\\partial L}{\\partial \\hat{y}} \\cdot \\frac{\\partial \\hat{y}}{\\partial z^{[2]}} \\cdot \\frac{\\partial z^{[2]}}{\\partial W^{[2]}}$$

This simplifies to:

$$\\frac{\\partial L}{\\partial W^{[2]}} = (\\hat{y} - y) \\cdot a^{[1]T}$$

## Gradient Descent Update

Once we have the gradients, we update the weights:

$$W := W - \\alpha \\frac{\\partial L}{\\partial W}$$

Where $\\alpha$ is the learning rate.

## Key Insights

1. **Computational Efficiency**: Backpropagation reuses computations from the forward pass
2. **Automatic Differentiation**: Modern frameworks like PyTorch handle this automatically
3. **Vanishing Gradients**: Deep networks can suffer from gradients becoming too small

## Code Example

\`\`\`python
import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# PyTorch handles backpropagation automatically!
model = SimpleNet(784, 128, 10)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Training step
output = model(input_data)
loss = criterion(output, target)
loss.backward()  # Backpropagation happens here
optimizer.step()  # Update weights
\`\`\`

## Conclusion

Understanding backpropagation is crucial for anyone working with neural networks. While modern frameworks abstract away the details, knowing the mathematics helps you debug issues, design better architectures, and understand why certain techniques work.

The beauty of backpropagation lies in its simplicity: it's just the chain rule applied systematically through a computational graph.
`
  },
  {
    id: "2",
    title: "Attention Mechanisms: The Foundation of Transformers and LLMs",
    date: "2024-12-10",
    readTime: "12 min read",
    category: "Large Language Models",
    excerpt: "Exploring how attention mechanisms revolutionized NLP and enabled the creation of powerful language models like GPT and BERT.",
    content: `
# Attention Mechanisms: The Foundation of Transformers and LLMs

The attention mechanism, introduced in the landmark paper "Attention Is All You Need," fundamentally changed how we approach sequence modeling and gave birth to modern large language models.

## The Problem with RNNs

Traditional RNNs process sequences sequentially, leading to:
- **Long-range dependency issues**: Information from early tokens gets diluted
- **Sequential processing**: Cannot be parallelized effectively
- **Fixed context**: Limited ability to focus on relevant parts

## Self-Attention: The Core Idea

Self-attention allows each token to attend to all other tokens in the sequence. The mechanism computes three vectors for each token:

- **Query (Q)**: What am I looking for?
- **Key (K)**: What do I contain?
- **Value (V)**: What information do I carry?

## Mathematical Formulation

The attention score between tokens $i$ and $j$ is computed as:

$$\\text{score}(q_i, k_j) = \\frac{q_i \\cdot k_j}{\\sqrt{d_k}}$$

Where $d_k$ is the dimension of the key vectors. The scaling factor $\\sqrt{d_k}$ prevents the dot products from becoming too large.

The attention weights are then:

$$\\alpha_{ij} = \\frac{\\exp(\\text{score}(q_i, k_j))}{\\sum_{k=1}^{n} \\exp(\\text{score}(q_i, k_k))}$$

Finally, the output is a weighted sum of values:

$$\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V$$

## Multi-Head Attention

Instead of a single attention mechanism, transformers use multiple attention heads in parallel:

$$\\text{MultiHead}(Q, K, V) = \\text{Concat}(\\text{head}_1, ..., \\text{head}_h)W^O$$

Where each head is:

$$\\text{head}_i = \\text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

This allows the model to attend to different aspects of the input simultaneously.

## Why It Works

| Feature | Benefit |
|---------|---------|
| Parallel Processing | Much faster training than RNNs |
| Long-range Dependencies | Direct connections between all tokens |
| Interpretability | Attention weights show what the model focuses on |
| Scalability | Enables training on massive datasets |

## Positional Encoding

Since attention has no inherent notion of order, we add positional encodings:

$$PE_{(pos, 2i)} = \\sin\\left(\\frac{pos}{10000^{2i/d}}\\right)$$
$$PE_{(pos, 2i+1)} = \\cos\\left(\\frac{pos}{10000^{2i/d}}\\right)$$

## Implementation Example

\`\`\`python
import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights
    
    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        
        # Linear projections and split into heads
        Q = self.W_q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply attention
        x, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # Final linear projection
        output = self.W_o(x)
        
        return output, attention_weights
\`\`\`

## Impact on LLMs

Attention mechanisms enabled:
- **GPT series**: Autoregressive language models with causal attention
- **BERT**: Bidirectional encoding with masked language modeling
- **T5**: Unified text-to-text framework
- **Modern LLMs**: Models with billions of parameters

## Conclusion

The attention mechanism is arguably the most important innovation in deep learning of the past decade. It's the foundation upon which all modern LLMs are built, enabling them to process and generate human-like text at unprecedented scale and quality.

Understanding attention is essential for anyone working with or building upon large language models.
`
  },
  {
    id: "3",
    title: "Convolutional Neural Networks: From Theory to Object Detection",
    date: "2024-12-01",
    readTime: "10 min read",
    category: "Computer Vision",
    excerpt: "Understanding how CNNs extract hierarchical features from images and power modern computer vision applications.",
    content: `
# Convolutional Neural Networks: From Theory to Object Detection

Convolutional Neural Networks (CNNs) revolutionized computer vision by learning hierarchical feature representations directly from raw pixels.

## The Convolution Operation

A convolution applies a filter (kernel) across an image to extract features. Mathematically:

$$(I * K)(i, j) = \\sum_{m}\\sum_{n} I(i-m, j-n) \\cdot K(m, n)$$

Where $I$ is the input image and $K$ is the kernel.

## Key Components

### 1. Convolutional Layers

Each filter learns to detect specific features:
- **Early layers**: Edges, corners, textures
- **Middle layers**: Object parts (eyes, wheels, windows)
- **Deep layers**: Complete objects and scenes

The output feature map size is:

$$O = \\frac{W - K + 2P}{S} + 1$$

Where:
- $W$ = input width
- $K$ = kernel size
- $P$ = padding
- $S$ = stride

### 2. Pooling Layers

Pooling reduces spatial dimensions while retaining important features:

$$\\text{MaxPool}(X) = \\max_{i,j \\in \\text{window}} X_{i,j}$$

Benefits:
- Reduces computational cost
- Provides translation invariance
- Controls overfitting

### 3. Activation Functions

**ReLU** is the most common:

$$\\text{ReLU}(x) = \\max(0, x)$$

**Leaky ReLU** addresses dying neurons:

$$\\text{LeakyReLU}(x) = \\begin{cases} x & \\text{if } x > 0 \\\\ \\alpha x & \\text{otherwise} \\end{cases}$$

## Classic CNN Architectures

| Architecture | Year | Key Innovation |
|--------------|------|----------------|
| LeNet | 1998 | First successful CNN |
| AlexNet | 2012 | Deep CNN with ReLU, dropout |
| VGG | 2014 | Very deep networks (16-19 layers) |
| ResNet | 2015 | Skip connections, 152+ layers |
| EfficientNet | 2019 | Compound scaling |

## Object Detection with CNNs

Modern object detection combines classification and localization:

### Region-based Methods (R-CNN family)

1. **R-CNN**: Selective search + CNN features
2. **Fast R-CNN**: ROI pooling for speed
3. **Faster R-CNN**: Region Proposal Network (RPN)

### Single-shot Methods

**YOLO (You Only Look Once)** divides the image into a grid and predicts:

$$P(\\text{class} | \\text{object}) \\times P(\\text{object}) \\times \\text{IoU}$$

For each grid cell.

## Implementation: Simple CNN

\`\`\`python
import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
        
        # Activation
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Conv block 1
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        
        # Conv block 2
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        
        # Conv block 3
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

# Usage
model = SimpleCNN(num_classes=10)
input_tensor = torch.randn(1, 3, 32, 32)  # Batch of 1, RGB image, 32x32
output = model(input_tensor)
print(f"Output shape: {output.shape}")  # [1, 10]
\`\`\`

## Advanced Techniques

### Data Augmentation

Essential for preventing overfitting:
- Random crops and flips
- Color jittering
- Cutout/Mixup
- AutoAugment

### Transfer Learning

Pre-trained models on ImageNet can be fine-tuned:

\`\`\`python
import torchvision.models as models

# Load pre-trained ResNet
resnet = models.resnet50(pretrained=True)

# Freeze early layers
for param in resnet.parameters():
    param.requires_grad = False

# Replace final layer
num_features = resnet.fc.in_features
resnet.fc = nn.Linear(num_features, num_classes)
\`\`\`

## Conclusion

CNNs remain the backbone of computer vision despite the rise of Vision Transformers. Their inductive biases (translation equivariance, local connectivity) make them highly efficient for image tasks.

Understanding CNNs is fundamental for anyone working in computer vision, from image classification to complex tasks like instance segmentation and 3D reconstruction.
`
  },
  {
    id: "4",
    title: "Eigenvalues and Eigenvectors: The Heart of Linear Algebra",
    date: "2024-11-20",
    readTime: "7 min read",
    category: "Mathematics",
    excerpt: "Exploring the geometric intuition and applications of eigenvalues and eigenvectors in machine learning and data science.",
    content: `
# Eigenvalues and Eigenvectors: The Heart of Linear Algebra

Eigenvalues and eigenvectors are among the most important concepts in linear algebra, with applications spanning from Principal Component Analysis to Google's PageRank algorithm.

## Definition

For a square matrix $A$, a non-zero vector $v$ is an **eigenvector** if:

$$Av = \\lambda v$$

Where $\\lambda$ is the corresponding **eigenvalue**.

This means that $A$ only scales $v$ by $\\lambda$, without changing its direction.

## Geometric Intuition

Consider a transformation matrix $A$. Most vectors change both magnitude AND direction when multiplied by $A$. Eigenvectors are special: they only get scaled.

For a 2D rotation matrix:

$$R_{\\theta} = \\begin{bmatrix} \\cos\\theta & -\\sin\\theta \\\\ \\sin\\theta & \\cos\\theta \\end{bmatrix}$$

If $\\theta \\neq 0, \\pi$, there are no real eigenvectors (every vector rotates).

## Finding Eigenvalues

To find eigenvalues, we solve the characteristic equation:

$$\\det(A - \\lambda I) = 0$$

For a $2 \\times 2$ matrix:

$$A = \\begin{bmatrix} a & b \\\\ c & d \\end{bmatrix}$$

The characteristic polynomial is:

$$\\lambda^2 - (a+d)\\lambda + (ad-bc) = 0$$

Using the quadratic formula:

$$\\lambda = \\frac{(a+d) \\pm \\sqrt{(a+d)^2 - 4(ad-bc)}}{2}$$

## Example Calculation

Let's find eigenvalues for:

$$A = \\begin{bmatrix} 4 & 1 \\\\ 2 & 3 \\end{bmatrix}$$

Characteristic equation:

$$\\det\\begin{bmatrix} 4-\\lambda & 1 \\\\ 2 & 3-\\lambda \\end{bmatrix} = 0$$

$$(4-\\lambda)(3-\\lambda) - 2 = 0$$
$$\\lambda^2 - 7\\lambda + 10 = 0$$
$$(\\lambda - 5)(\\lambda - 2) = 0$$

So $\\lambda_1 = 5$ and $\\lambda_2 = 2$.

For $\\lambda_1 = 5$:

$$(A - 5I)v = 0$$
$$\\begin{bmatrix} -1 & 1 \\\\ 2 & -2 \\end{bmatrix}\\begin{bmatrix} v_1 \\\\ v_2 \\end{bmatrix} = 0$$

This gives $v_1 = \\begin{bmatrix} 1 \\\\ 1 \\end{bmatrix}$.

## Key Properties

1. **Trace**: $\\text{tr}(A) = \\sum \\lambda_i$
2. **Determinant**: $\\det(A) = \\prod \\lambda_i$
3. **Symmetric matrices**: Always have real eigenvalues and orthogonal eigenvectors
4. **Positive definite**: All eigenvalues are positive

## Diagonalization

If $A$ has $n$ linearly independent eigenvectors, we can write:

$$A = PDP^{-1}$$

Where:
- $P$ = matrix of eigenvectors
- $D$ = diagonal matrix of eigenvalues

This makes computing powers of $A$ easy:

$$A^k = PD^kP^{-1}$$

## Applications in Machine Learning

### 1. Principal Component Analysis (PCA)

PCA finds eigenvectors of the covariance matrix:

$$C = \\frac{1}{n}X^TX$$

The eigenvectors with largest eigenvalues are the principal components.

### 2. PageRank

Google's PageRank finds the dominant eigenvector of the web graph's adjacency matrix.

### 3. Spectral Clustering

Uses eigenvectors of the graph Laplacian:

$$L = D - A$$

Where $D$ is the degree matrix and $A$ is the adjacency matrix.

## Python Implementation

\`\`\`python
import numpy as np

# Define matrix
A = np.array([[4, 1],
              [2, 3]])

# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)

print("Eigenvalues:", eigenvalues)
print("Eigenvectors:\\n", eigenvectors)

# Verify: Av = λv
v1 = eigenvectors[:, 0]
lambda1 = eigenvalues[0]

print("\\nVerification:")
print("Av =", A @ v1)
print("λv =", lambda1 * v1)

# Diagonalization
P = eigenvectors
D = np.diag(eigenvalues)
P_inv = np.linalg.inv(P)

A_reconstructed = P @ D @ P_inv
print("\\nReconstructed A:\\n", A_reconstructed)
\`\`\`

## Spectral Theorem

For symmetric matrices $A = A^T$:

$$A = Q\\Lambda Q^T$$

Where:
- $Q$ is orthogonal ($Q^TQ = I$)
- $\\Lambda$ is diagonal with eigenvalues

This is fundamental to many optimization algorithms.

## Conclusion

Eigenvalues and eigenvectors provide deep insights into linear transformations. They're not just abstract mathematical concepts but practical tools used daily in:

- Dimensionality reduction (PCA)
- Graph algorithms (PageRank, spectral clustering)
- Differential equations
- Quantum mechanics
- Stability analysis

Mastering these concepts opens doors to understanding advanced topics in machine learning, optimization, and scientific computing.
`
  },
  {
    id: "5",
    title: "Building Scalable Backend Systems: Microservices and Message Queues",
    date: "2024-11-10",
    readTime: "11 min read",
    category: "Backend Architecture",
    excerpt: "A practical guide to designing distributed systems that scale, focusing on microservices patterns and asynchronous communication.",
    content: `
# Building Scalable Backend Systems: Microservices and Message Queues

As applications grow, monolithic architectures become bottlenecks. This post explores how to design scalable backend systems using microservices and message queues.

## The Monolith Problem

Traditional monolithic applications have several limitations:

- **Scaling**: Must scale the entire application, even if only one component needs more resources
- **Deployment**: Small changes require redeploying everything
- **Technology lock-in**: Entire codebase uses the same tech stack
- **Team coordination**: Multiple teams working on the same codebase

## Microservices Architecture

Microservices decompose applications into small, independent services that:

1. Own their data
2. Communicate via APIs
3. Can be deployed independently
4. Use the best tool for their specific job

### Service Boundaries

Use Domain-Driven Design (DDD) to identify service boundaries:

\`\`\`
User Service
├── Authentication
├── Profile Management
└── User Preferences

Order Service
├── Order Creation
├── Order Processing
└── Order History

Payment Service
├── Payment Processing
├── Refunds
└── Transaction History
\`\`\`

## Communication Patterns

### 1. Synchronous (REST/gRPC)

**REST API Example:**

\`\`\`python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

class Order(BaseModel):
    user_id: int
    product_id: int
    quantity: int

@app.post("/orders")
async def create_order(order: Order):
    # Validate user exists (call User Service)
    user = await get_user(order.user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Process payment (call Payment Service)
    payment_result = await process_payment(order)
    
    # Create order
    order_id = await save_order(order)
    
    return {"order_id": order_id, "status": "created"}
\`\`\`

**Problems with synchronous communication:**
- Tight coupling
- Cascading failures
- Increased latency

### 2. Asynchronous (Message Queues)

Message queues decouple services and improve resilience.

## Message Queue Patterns

### Publish/Subscribe

Multiple consumers receive the same message:

\`\`\`python
import pika
import json

# Publisher
def publish_order_created(order_data):
    connection = pika.BlockingConnection(
        pika.ConnectionParameters('localhost')
    )
    channel = connection.channel()
    
    channel.exchange_declare(
        exchange='orders',
        exchange_type='fanout'
    )
    
    message = json.dumps(order_data)
    channel.basic_publish(
        exchange='orders',
        routing_key='',
        body=message
    )
    
    connection.close()

# Subscriber 1: Email Service
def send_order_confirmation():
    connection = pika.BlockingConnection(
        pika.ConnectionParameters('localhost')
    )
    channel = connection.channel()
    
    channel.exchange_declare(
        exchange='orders',
        exchange_type='fanout'
    )
    
    result = channel.queue_declare(queue='', exclusive=True)
    queue_name = result.method.queue
    
    channel.queue_bind(
        exchange='orders',
        queue=queue_name
    )
    
    def callback(ch, method, properties, body):
        order = json.loads(body)
        send_email(order['user_email'], order)
    
    channel.basic_consume(
        queue=queue_name,
        on_message_callback=callback,
        auto_ack=True
    )
    
    channel.start_consuming()

# Subscriber 2: Inventory Service
def update_inventory():
    # Similar setup, different callback
    def callback(ch, method, properties, body):
        order = json.loads(body)
        reduce_inventory(order['product_id'], order['quantity'])
    
    # ... rest of setup
\`\`\`

### Work Queue

Tasks distributed among workers:

\`\`\`python
# Producer
def queue_image_processing(image_url):
    channel.basic_publish(
        exchange='',
        routing_key='image_processing',
        body=image_url,
        properties=pika.BasicProperties(
            delivery_mode=2,  # Persistent
        )
    )

# Consumer (multiple workers can run)
def process_images():
    def callback(ch, method, properties, body):
        image_url = body.decode()
        
        # Heavy processing
        process_image(image_url)
        
        ch.basic_ack(delivery_tag=method.delivery_tag)
    
    channel.basic_qos(prefetch_count=1)  # Fair dispatch
    channel.basic_consume(
        queue='image_processing',
        on_message_callback=callback
    )
    
    channel.start_consuming()
\`\`\`

## Database Strategies

### Database per Service

Each service owns its database:

\`\`\`
┌─────────────────┐     ┌─────────────────┐
│  User Service   │     │  Order Service  │
├─────────────────┤     ├─────────────────┤
│   User DB       │     │   Order DB      │
└─────────────────┘     └─────────────────┘
\`\`\`

**Benefits:**
- Service independence
- Technology flexibility
- Easier scaling

**Challenges:**
- No ACID transactions across services
- Data consistency requires patterns like Saga

### Saga Pattern

Distributed transactions as a sequence of local transactions:

\`\`\`python
class OrderSaga:
    async def execute(self, order_data):
        try:
            # Step 1: Reserve inventory
            inventory_reserved = await inventory_service.reserve(
                order_data['product_id'],
                order_data['quantity']
            )
            
            # Step 2: Process payment
            payment_processed = await payment_service.charge(
                order_data['user_id'],
                order_data['amount']
            )
            
            # Step 3: Create order
            order_created = await order_service.create(order_data)
            
            return order_created
            
        except InventoryException:
            # No compensation needed
            raise
            
        except PaymentException:
            # Compensate: Release inventory
            await inventory_service.release(
                order_data['product_id'],
                order_data['quantity']
            )
            raise
            
        except OrderException:
            # Compensate: Refund and release inventory
            await payment_service.refund(order_data['user_id'])
            await inventory_service.release(
                order_data['product_id'],
                order_data['quantity']
            )
            raise
\`\`\`

## Monitoring and Observability

### Distributed Tracing

Track requests across services:

\`\`\`python
from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

tracer = trace.get_tracer(__name__)

@app.post("/orders")
async def create_order(order: Order):
    with tracer.start_as_current_span("create_order"):
        with tracer.start_as_current_span("validate_user"):
            user = await get_user(order.user_id)
        
        with tracer.start_as_current_span("process_payment"):
            payment = await process_payment(order)
        
        return {"order_id": order_id}
\`\`\`

### Metrics

Key metrics to track:

| Metric | Description |
|--------|-------------|
| Request Rate | Requests per second |
| Error Rate | Failed requests percentage |
| Latency | Response time (p50, p95, p99) |
| Queue Depth | Messages waiting to be processed |

## Best Practices

1. **API Versioning**: Use versioned endpoints (\`/v1/orders\`)
2. **Circuit Breakers**: Prevent cascading failures
3. **Retry Logic**: With exponential backoff
4. **Idempotency**: Handle duplicate messages gracefully
5. **Health Checks**: Implement \`/health\` endpoints
6. **Rate Limiting**: Protect services from overload

## Conclusion

Building scalable backend systems requires careful consideration of:

- Service boundaries and responsibilities
- Communication patterns (sync vs async)
- Data management strategies
- Failure handling and resilience
- Monitoring and observability

Microservices and message queues are powerful tools, but they add complexity. Start with a monolith and evolve to microservices as your needs grow.

The key is understanding the trade-offs and choosing the right architecture for your specific requirements.
`
  }
];
