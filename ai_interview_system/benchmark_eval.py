import time
import json
import ollama

history = [
    ("Explain the difference between process and thread.", "A process is an execution of a program with its own memory space, whereas a thread is a lightweight unit of execution within a process that shares memory with other threads."),
    ("[Skipped by candidate]", "[Skipped by candidate]"),
    ("[Skipped by candidate]", "[Skipped by candidate]"),
    ("What is RESTful API design?", "REST APIs use HTTP methods like GET, POST, PUT, DELETE to perform CRUD operations on resources identified by URIs. They are stateless."),
    ("[Skipped by candidate]", "[Skipped by candidate]"),
    ("[Skipped by candidate]", "[Skipped by candidate]"),
    ("[Skipped by candidate]", "[Skipped by candidate]"),
    ("[Skipped by candidate]", "[Skipped by candidate]"),
    ("[Skipped by candidate]", "[Skipped by candidate]"),
    ("What is a deadlock? Name the four conditions that must hold for one to occur.", "At the at-lock is a state in computing where two or more processes are stuck waiting forever because each needs a resource that the other process currently holds.")
]

compiled_log = ""
for i, (q, a) in enumerate(history):
    compiled_log += f"\n[Q{i+1}]: {q}\n[Answer {i+1}]: {a}\n---"

prompt = f"""You are a technical interviewer evaluating a candidate across {len(history)} questions.
Score each question from 0 to 10 (0 if skipped or blank).
Interview transcript:
{compiled_log}

Output ONLY valid JSON matching this schema:
{{
  "per_question_scores": [7, 0, 0, 8, 0, 0, 0, 0, 0, 7],
  "per_question_feedback": ["Good definition", "Skipped", "Skipped", "Clear explanation", "Skipped", "Skipped", "Skipped", "Skipped", "Skipped", "Defined deadlock correctly but missed naming mutual exclusion, hold and wait, no preemption, and circular wait"],
  "score_breakdown": {{
    "technical_correctness": 65,
    "reasoning_approach": 60,
    "completeness_relevance": 55,
    "answer_format_quality": 50,
    "communication_clarity": 65
  }},
  "technical_strengths": "Strong grasp of OS process fundamentals and core deadlock definition.",
  "critical_gaps_identified": "Missed the four Coffman conditions for deadlocks (mutual exclusion, hold & wait, no preemption, circular wait). Several questions were skipped.",
  "communication_feedback": "Spoken answers are clear and concise. Ensure comprehensive answers covering all parts of multi-part questions.",
  "overall_verdict": "Lean Hire",
  "recommended_study_topics": ["Deadlock Coffman Conditions", "Concurrency & Semaphores", "System Architecture", "Database Transactions", "Networking Protocols"],
  "learning_resources": ["Operating System Concepts (Silberschatz)", "Designing Data-Intensive Applications", "LeetCode Concurrency"],
  "improvement_plan": "Master multi-part conceptual questions by addressing each clause systematically. Review operating systems deadlock prevention algorithms."
}}"""

t0 = time.time()
try:
    print("Calling phi3...")
    client = ollama.Client(timeout=180)
    resp = client.generate(model='phi3', prompt=prompt, options={'temperature': 0.1, 'num_predict': 900})
    raw = resp['response'].strip()
    print("Response time:", round(time.time() - t0, 2), "s")
    print("Raw output length:", len(raw))
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1:
        data = json.loads(raw[start:end+1])
        print("SUCCESS! Parsed JSON:")
        print("Verdict:", data.get("overall_verdict"))
        print("Scores:", data.get("per_question_scores"))
        print("Strengths:", data.get("technical_strengths"))
    else:
        print("Could not find JSON brackets in:", raw)
except Exception as e:
    print("Error:", e)
