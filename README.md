# AdobeDXLaunchpad
A GenAI Assistant for Adobe's Digital Certificate Website

Adobe’s Digital Experience (DX) Platform has an important, ambitious goal of making knowledge accessible and empowering learners by offering high-demand industry-relevant skills to students and professionals alike.  With an extensive portfolio of more than 52 certifications and 23 courses alongside supplemental resources, the platform offers something for everyone. However, to truly reach and benefit everyone - professionals, students, career changers, and lifelong learners alike - the platform must deliver a seamless and personalized user experience that helps each individual see how Adobe’s offerings can empower them to achieve their unique goals. To accomplish this, we must address several inherent challenges.

Key Challenges:

- The current platform needs to adequately guide users to identify the products that are right for them, rather than relying on users to explore the plethora of options manually 

- As Adobe continues to expand its offerings with new products, certifications, and courses, the current system must evolve to scale efficiently

- The platform needs a tailored approach to create learning trajectories, making it easier for users to connect offerings to their aspirations

- The platform should foster inclusivity and engagement by recommending resources that align with a student’s diverse intellectual background and current knowledge

# Adobe Course Dependency Graph (RAG Pipeline)

This project is a **Retrieval-Augmented Generation (RAG) pipeline** that processes Adobe courses and certificates, stores them in a ChromaDB database, and generates responses and course trajectories based on user input. 

It includes a **Flask API backend** and a **React (Next.js) frontend**, allowing users to interact with the system through a web interface.

---

## 🚀 **Getting Started**

### **1️⃣ Set Up the Environment**
#### **🔹 Install Conda (If Not Installed)**
If you don’t have Conda installed, download and install **Miniconda** or **Anaconda**:
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- [Anaconda](https://www.anaconda.com/products/distribution)

#### **🔹 Create and Activate the Conda Environment**
Use the provided `environment.yml` file to create the environment:
```bash
conda env create -f environment.yml
conda activate RAG
```
### **2️⃣ Set Up the Backend**
**🔹 Navigate to the Backend Directory**
```
cd dependency_graph
```
**🔹 Run the API**
```
python api.py
```
This should start the Flask server at:
```
http://127.0.0.1:5000
```
### **3️⃣ Setup the Frontend (Next.js React App)**

**In a new terminal, Navigate to the Frontend Directory**
```
cd chatbot
```
**Install frontend dependencies**
```
npm install
```
**Build the Next.js App**
```
npm run build
```
**Start the Frontend**
```
npm start
```
**Open the frontend**  
Visit:
```
http://localhost:3000
```
And feel free to interact with the chatbot!
