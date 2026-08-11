from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader,TextLoader
from dotenv import load_dotenv
import os
load_dotenv()

nvidia_api= os.getenv("NVIDIA_API_KEY")
llm_model= os.getenv("NVIDIA_LLM_MODEL")

llm=ChatNVIDIA(
    model=nvidia_api,
    api_key= nvidia_api,
    temperature=0.33
)
loader=PyPDFLoader(file_path=input("Paste the path of the file: "))
# loader=PyPDFLoader("30_05_2026\sample.pdf")
result=loader.load()
resume_text=result[0].page_content
jd_loader=TextLoader(file_path=input("Enter the file path of jd in txt format: "))
jd_result=jd_loader.load()
jd_text=jd_result[0].page_content

template = """
You are an expert ATS (Applicant Tracking System) resume reviewer and hiring manager with 10 years of recruitment experience.
Analyze the resume against the provided job description.
Resume:
{resume}
Job Description:
{jd}
Tasks:
1. Calculate ATS Match Score out of 100.
2. List matching skills found in both resume and JD.
3. List missing skills from the JD.
4. Identify important keywords missing from the resume.
5. Evaluate:
   - Technical Skills Match
   - Experience Match
   - Education Match
   - Project Relevance
6. Give detailed reasons for the score.
7. Suggest specific improvements to increase the ATS score.
8. Return the output in a structured format.
Output Format:
ATS Score: X/100
Matching Skills:
Missing Skills:
Strengths:
Weaknesses:
Recommendations:
Final Verdict:
"""
prompt=PromptTemplate.from_template(template)
final_prompt=prompt.format(resume=resume_text,jd=jd_text)
for chunk in llm.stream(final_prompt):
    print(chunk.content, end="")
