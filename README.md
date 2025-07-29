# Nivara

In today's digital era, misinformation poses a significant threat, influencing decisions and shaping global opinions with often inaccurate or deceptive content. **Nivara** is a solution designed to combat this issue by creating a reliable platform that leverages the combined power of **academic expertise** and **AI-driven models** to provide verified, fact-checked information.

## Table of Contents

- [About Nivara](#about-Nivara)
- [Key Features](#key-features)
  - [1. Expert Verification](#1-expert-verification)
  - [2. AI-Powered Analysis](#2-ai-powered-analysis)
  - [3. Incentivized Participation](#3-incentivized-participation)
  - [4. User-Friendly Interface](#4-user-friendly-interface)
  - [5. Scalability](#5-scalability)
- [Installation](#installation)
- [Contributing](#contributing)
- [License](#license)

## About Nivara

**Nivara** is a misinformation detection and verification system. It offers a platform where **academic scholars** verify the accuracy of articles and other media. By utilizing **Natural Language Processing (NLP)** and **Artificial Intelligence (AI)**, the app also automates the initial analysis of content, flagging it as potentially truthful or misleading. This multi-layered approach aims to ensure the reliability of information consumed by users.

## Key Features

### 1. Expert Verification

Nivara integrates **subject-matter expertise** as a core part of its fact-checking process. **Qualified scholars** review articles based on their expertise, assessing the truthfulness of the content.

- Scholars are matched to articles based on their domain knowledge (e.g., medical articles are reviewed by doctors or medical researchers).
- Verified experts provide a confidence score after reviewing an article, which is displayed alongside the content.
  
### 2. AI-Powered Analysis

Nivara employs state-of-the-art **NLP models** to assist with article evaluation. These models perform an initial pass to flag questionable content:

- NLP algorithms assess linguistic cues and cross-reference available data to detect bias, inconsistencies, or misinformation.
- AI models assign a probability score that indicates whether the content is likely to be true or false, which is then reviewed by scholars for a more detailed examination.

### 3. Incentivized Participation

Scholars are incentivized to contribute by being rewarded for accurate and thorough reviews. Features include:

- **Reward Mechanism:** Scholars earn points or credits based on the quality and frequency of their reviews.
- **Performance-Based Incentives:** A reputation system ensures that scholars with a history of accurate reviews are prioritized for more content and higher rewards.

### 4. User-Friendly Interface

Nivara provides an intuitive, clean interface where users can:

- Instantly see the **credibility score** of articles.
- Access detailed **in-depth analysis** (via a premium subscription).
- Interact with visualizations that show how the reliability of an article changes over time as more reviews and analysis are conducted.

### 5. Scalability

Nivara is built for future expansion, allowing for additional media types to be verified:

- **Images and Videos:** Upcoming features will enable image and video fact-checking, addressing visual misinformation and deepfakes.
- The system is designed to handle an increasing number of users, reviews, and content types without sacrificing performance.

## Installation

To run this project locally, follow these steps:

1. Clone the repository:

    ```bash
    git clone https://github.com/Devonkedev/Niva.git
    ```

2. Navigate into the project directory:

    ```bash
    cd Niva
    ```

3. Install dependencies for both frontend and backend:

    ```bash
    # Install server dependencies
    cd backend
    npm install
    
    # Install client dependencies
    cd ../frontend
    npm install
    ```

4. Run the application:

    ```bash
    # Run server
    cd backend
    npm start
    
    # Run client
    cd ../frontend
    npm start
    ```

5. Visit `http://localhost:3000` in your browser to access the app.

## Contributing

Contributions are welcome! If you'd like to contribute to the project:

1. Fork the repository.
2. Create a new branch with your feature or bug fix:
    
    ```bash
    git checkout -b feature-name
    ```

3. Commit your changes:

    ```bash
    git commit -m "Your message"
    ```

4. Push to your branch:

    ```bash
    git push origin feature-name
    ```

5. Open a Pull Request, detailing your changes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
