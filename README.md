# Introduction to Soft Matter 2025

Welcome to the course repository for Introduction to Soft Matter 2025! This repository contains course materials, assignments, and collaborative learning resources.

**Repository**: [https://github.com/comphy-lab/Intro-Soft-Matter-2025](https://github.com/comphy-lab/Intro-Soft-Matter-2025)

---

## Take-Home Homework Assignment: Contact Line Singularity

### Problem Statement

In this assignment, you will solve a third-order ordinary differential equation (ODE) that describes the contact line singularity in fluid mechanics:

$$\frac{d^3h}{dx^3} = -0.01\left(\frac{1}{h^2+h}\right)$$

**Boundary Conditions:**
- $h(0) = 0$
- $h'(0) = 1$
- $h''(\infty) = 0$

Your task is to numerically solve this ODE and visualize the solution by plotting:
1. $h'(x)$ vs $x$ (first subplot)
2. $h''(x)$ vs $x$ (second subplot)

### Assignment Objectives

This assignment is designed to help you:
1. **Experience AI-assisted coding**: Learn how to effectively use Large Language Models (LLMs) for code generation
2. **Develop debugging skills**: Understand that initial AI-generated code often requires refinement
3. **Practice iterative development**: Learn to improve code through testing and iteration
4. **Collaborate using GitHub**: Gain experience with version control and pull requests

### Part 1: Initial LLM-Generated Code

#### Step 1: Choose Your Tools
- Select any LLM of your choice (ChatGPT, Claude, Gemini, etc.)
- Choose any programming language you prefer (Python, MATLAB, Julia, etc.)

#### Step 2: Use This Prompt

Copy and paste the following prompt into your chosen LLM:

```text
Solve the following ODE with boundary conditions:
- ODE: dh/dx = -0.01(1/(h^2+h))
- BCs: h(0) = 0, h'(0) = 1, h''() = 0

Write a [INSERT YOUR LANGUAGE HERE] script that:
1. Solves this ODE numerically
2. Plots h'(x) vs x in the first subplot
3. Plots h''(x) vs x in the second subplot

Please provide complete, runnable code.
```

#### Step 3: Save the Initial Code

- Save exactly what the LLM gives you as `firstname-lastname-v1.py` (or `.m`, `.jl`, etc., depending on your language)
- **Do not modify this code yet!**
- This file represents your first attempt using AI assistance

#### What to Expect

Most likely, the initial code will have issues! This is completely normal and expected. Common problems include:
- Incorrect boundary condition implementation
- Numerical instability
- Plotting errors
- Missing imports or dependencies

**This is part of the learning process!** We want to see both the initial attempt and your refined solution.

### Part 2: Refined Working Solution

Now it's time to make the code actually work!

#### Debugging Strategies

You can use any combination of these approaches:
- Continue conversing with the LLM, explaining what went wrong
- Try different LLMs to compare approaches
- Search online for similar problems and solutions
- Discuss with classmates (collaboration is encouraged!)
- Read documentation for numerical ODE solvers
- Experiment with different numerical methods

#### Step 4: Create Working Code

- Debug, refine, and improve the code until it successfully:
  - Solves the ODE with the correct boundary conditions
  - Produces meaningful plots
  - Runs without errors
- Save your final working version as `firstname-lastname-v2.py` (or appropriate extension)

#### What We're Looking For

Your refined code should:
- Successfully solve the ODE numerically
- Properly handle the boundary condition at infinity
- Generate clear, labeled plots
- Include comments explaining key steps
- Run successfully without modifications

---

## Submission Instructions

You will submit your work by creating a Pull Request (PR) on GitHub. Below are two methods: GUI-based (recommended for beginners) and terminal-based (for those comfortable with Git).

### Method 1: GUI-Based Submission (Recommended)

#### Step 1: Fork the Repository

1. Go to [https://github.com/comphy-lab/Intro-Soft-Matter-2025](https://github.com/comphy-lab/Intro-Soft-Matter-2025)
2. Click the **"Fork"** button in the top-right corner
3. This creates a copy of the repository under your GitHub account

#### Step 2: Upload Your Files

1. Navigate to your forked repository (it will be at `https://github.com/YOUR-USERNAME/Intro-Soft-Matter-2025`)
2. Click **"Add file"** � **"Upload files"**
3. Drag and drop or select both files:
   - `firstname-lastname-v1.py` (or your language extension)
   - `firstname-lastname-v2.py`
4. Scroll down to the "Commit changes" section

#### Step 3: Write a Commit Message

In the commit message box, write something meaningful like:

```
Add contact line singularity homework submission

- Initial LLM-generated code (v1)
- Refined working solution (v2)
```

**Why commit messages matter**: Good commit messages help collaborators (and your future self!) understand what changes were made and why. They're an essential part of professional software development and scientific computing.

#### Step 4: Create Pull Request

1. After committing, you'll see a banner saying **"This branch is 1 commit ahead of comphy-lab:main"**
2. Click **"Contribute"** � **"Open pull request"**
3. In the PR description, briefly explain:
   - Which LLM you used
   - What challenges you encountered
   - How you solved them
   - Any insights you gained from the process

#### Step 5: Submit

Click **"Create pull request"** - that's your submission!

### Method 2: Terminal-Based Submission (Advanced)

For those comfortable with Git and the command line:

#### Step 1: Fork and Clone

```bash
# First, fork the repository via the GitHub web interface
# Then clone your fork:
git clone https://github.com/YOUR-USERNAME/Intro-Soft-Matter-2025.git
cd Intro-Soft-Matter-2025
```

#### Step 2: Create a Branch

```bash
# Create a new branch for your submission
git checkout -b homework-submission-yourname
```

#### Step 3: Add Your Files

```bash
# Copy your files to the repository directory
cp /path/to/your/firstname-lastname-v1.py .
cp /path/to/your/firstname-lastname-v2.py .

# Stage the files
git add firstname-lastname-v1.py firstname-lastname-v2.py
```

#### Step 4: Commit with a Good Message

```bash
git commit -m "Add contact line singularity homework submission

- Initial LLM-generated code (v1)
- Refined working solution (v2)
- Used [LLM name] for initial generation
- Debugged numerical boundary condition handling"
```

#### Step 5: Push and Create PR

```bash
# Push to your fork
git push origin homework-submission-yourname
```

Then go to GitHub and click "Compare & pull request" to open your PR.

---

## Important Notes

### On Commit Messages

Writing clear, descriptive commit messages is a crucial skill in collaborative development:
- **What changed**: Briefly describe the modifications
- **Why it changed**: Explain the motivation or problem being solved
- **How it changed**: If not obvious, mention the approach taken

Good commit messages create a valuable project history that helps everyone understand the evolution of the codebase.

### On Experimentation and Failure

- **Embrace initial failures**: The gap between v1 and v2 shows your learning process
- **Document your journey**: In your PR, share what you learned
- **Ask questions**: Use the PR discussion to ask questions or share insights
- **Iterate**: You can push additional commits to your PR if you want to improve further

### On Programming Languages

- **Any language is acceptable**: Python, MATLAB, Julia, R, Mathematica, etc.
- **Choose what you know**: Or use this as an opportunity to learn something new!
- **Note your choice**: Mention your language choice in the PR description

### On Collaboration

- **Discussing is encouraged**: Talk with classmates about approaches
- **Your code should be your own**: You can discuss strategies, but write your own code
- **Cite your sources**: If you use external resources or code snippets, acknowledge them
- **Learn from each other**: Review other students' PRs to see different approaches

---

## Questions?

If you have questions about the assignment or encounter technical difficulties with GitHub:
- Open an issue in the repository
- Reach out during office hours
- Ask in the course discussion forum

Good luck, and happy coding! =D
