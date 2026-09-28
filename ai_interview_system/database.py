import sqlite3
import random

# ============================================================
# SINGLE SOURCE OF TRUTH — app.py imports these same lists
# ============================================================
COMPANIES_LIST = [
    "Google", "Microsoft", "Amazon", "Meta", "Apple",
    "Netflix", "Uber", "Adobe", "Oracle", "IBM",
    "TCS", "Infosys", "Wipro", "HCL", "Zoho",
    "Flipkart", "Swiggy", "Paytm", "Capgemini", "Accenture"
]

ROLES_LIST = [
    "Software Engineer (SDE-1)",
    "Software Engineer (SDE-2)",
    "Data Scientist (AI/ML Core)",
    "Site Reliability Engineer (SRE)",
    "Systems Infrastructure Architect",
    "Backend Developer",
    "Full Stack Developer",
    "ML Engineer",
    "DevOps Engineer",
    "Data Engineer",
]

# ============================================================
# QUESTION POOL — redesigned for 2-3 MINUTE answers
# Each question targets ONE single concept, not multiple.
# subject field also tags the intended answer style so the
# UI/evaluator can hint the candidate toward Voice/Text,
# Pseudocode, or Diagram mode.
#
# DSA Track (CTCI): 4 conceptual, 4 pseudocode, 2 diagram
# OS Track (Dinosaur Book): 5 conceptual, 5 diagram
# ============================================================
QUESTIONS_POOL = [
    # ---------------------------------------------------------
    # DSA TRACK — CONCEPTUAL (voice/text friendly, 2-3 min)
    # ---------------------------------------------------------
    ("DSA Track - Conceptual",
     "Cracking the Coding Interview (CTCI) by Gayle Laakmann McDowell",
     "What does it mean for an algorithm to run in O(1) amortized time? Give a simple example."),
    ("DSA Track - Conceptual",
     "Cracking the Coding Interview (CTCI) by Gayle Laakmann McDowell",
     "What is the difference between a Stack and a Queue? Give one real-world use case for each."),
    ("DSA Track - Conceptual",
     "Cracking the Coding Interview (CTCI) by Gayle Laakmann McDowell",
     "What is memoization, and how does it speed up recursive solutions like Fibonacci?"),
    ("DSA Track - Conceptual",
     "Cracking the Coding Interview (CTCI) by Gayle Laakmann McDowell",
     "What is the difference between a HashMap and a HashSet, and when would you use one over the other?"),

    # ---------------------------------------------------------
    # DSA TRACK — PSEUDOCODE friendly (2-3 min)
    # ---------------------------------------------------------
    ("DSA Track - Pseudocode",
     "Cracking the Coding Interview (CTCI) by Gayle Laakmann McDowell",
     "Write pseudocode to reverse a singly linked list."),
    ("DSA Track - Pseudocode",
     "Cracking the Coding Interview (CTCI) by Gayle Laakmann McDowell",
     "Write pseudocode for binary search on a sorted array."),
    ("DSA Track - Pseudocode",
     "Cracking the Coding Interview (CTCI) by Gayle Laakmann McDowell",
     "Write pseudocode to check if a string is a palindrome."),
    ("DSA Track - Pseudocode",
     "Cracking the Coding Interview (CTCI) by Gayle Laakmann McDowell",
     "Write pseudocode to find the maximum element in an unsorted array in a single pass."),

    # ---------------------------------------------------------
    # DSA TRACK — DIAGRAM friendly (2-3 min)
    # ---------------------------------------------------------
    ("DSA Track - Diagram",
     "Cracking the Coding Interview (CTCI) by Gayle Laakmann McDowell",
     "Draw a simple Binary Search Tree with 7 nodes and briefly explain how a search would traverse it."),
    ("DSA Track - Diagram",
     "Cracking the Coding Interview (CTCI) by Gayle Laakmann McDowell",
     "Draw a small graph with 5 nodes and explain the difference between BFS and DFS traversal on it."),

    # ---------------------------------------------------------
    # OS TRACK — CONCEPTUAL (voice/text friendly, 2-3 min)
    # ---------------------------------------------------------
    ("OS Track - Conceptual",
     "Operating System Concepts (\"The Dinosaur Book\") by Abraham Silberschatz, Peter Baer Galvin, and Greg Gagne",
     "What is the difference between a process and a thread?"),
    ("OS Track - Conceptual",
     "Operating System Concepts (\"The Dinosaur Book\") by Abraham Silberschatz, Peter Baer Galvin, and Greg Gagne",
     "What is a deadlock? Name the four conditions that must hold for one to occur."),
    ("OS Track - Conceptual",
     "Operating System Concepts (\"The Dinosaur Book\") by Abraham Silberschatz, Peter Baer Galvin, and Greg Gagne",
     "What is the difference between a Mutex and a Semaphore?"),
    ("OS Track - Conceptual",
     "Operating System Concepts (\"The Dinosaur Book\") by Abraham Silberschatz, Peter Baer Galvin, and Greg Gagne",
     "What is virtual memory, and why do operating systems use it instead of direct physical memory access?"),
    ("OS Track - Conceptual",
     "Operating System Concepts (\"The Dinosaur Book\") by Abraham Silberschatz, Peter Baer Galvin, and Greg Gagne",
     "What is the difference between preemptive and non-preemptive CPU scheduling?"),

    # ---------------------------------------------------------
    # OS TRACK — DIAGRAM friendly (2-3 min)
    # ---------------------------------------------------------
    ("OS Track - Diagram",
     "Operating System Concepts (\"The Dinosaur Book\") by Abraham Silberschatz, Peter Baer Galvin, and Greg Gagne",
     "Draw the process state diagram (New, Ready, Running, Waiting, Terminated) and explain one transition between two states."),
    ("OS Track - Diagram",
     "Operating System Concepts (\"The Dinosaur Book\") by Abraham Silberschatz, Peter Baer Galvin, and Greg Gagne",
     "Draw a simple diagram showing how virtual addresses are translated to physical addresses using a page table."),
    ("OS Track - Diagram",
     "Operating System Concepts (\"The Dinosaur Book\") by Abraham Silberschatz, Peter Baer Galvin, and Greg Gagne",
     "Draw a timeline diagram showing how Round Robin scheduling executes 3 processes with equal time slices."),
    ("OS Track - Diagram",
     "Operating System Concepts (\"The Dinosaur Book\") by Abraham Silberschatz, Peter Baer Galvin, and Greg Gagne",
     "Draw a simple memory layout diagram showing the difference between internal and external fragmentation."),
    ("OS Track - Diagram",
     "Operating System Concepts (\"The Dinosaur Book\") by Abraham Silberschatz, Peter Baer Galvin, and Greg Gagne",
     "Draw a diagram of the Producer-Consumer problem showing the shared buffer between two processes."),
]


def init_db():
    conn = sqlite3.connect("interview_system.db")
    cursor = conn.cursor()

    cursor.execute("DROP TABLE IF EXISTS company_roles")
    cursor.execute("DROP TABLE IF EXISTS question_bank")

    cursor.execute("""
        CREATE TABLE company_roles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            company_name TEXT,
            role_name TEXT,
            required_skills TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE question_bank (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            company_name TEXT,
            role_name TEXT,
            subject TEXT,
            source_reference TEXT,
            question TEXT
        )
    """)

    companies_data = [
        ("Google", "Software Engineer (SDE-1)", "data structures, algorithms, python, java, c++, dsa"),
        ("Microsoft", "Cloud Architecture Specialist", "operating systems, distributed systems, azure, networks"),
        ("Amazon", "SDE-II Backend Engineering", "java, distributed databases, scale, system design, dsa"),
        ("Meta", "Product Systems Engineer", "algorithms, parsing, performance infrastructure, oops"),
        ("Apple", "Firmware Kernel Engineer", "c programming, operating systems, embedded systems, low-level"),
        ("Netflix", "Data Platform Core Architect", "databases, distributed processing, sql, concurrency"),
        ("Uber", "Distributed Scalability Engineer", "concurrency, networking, algorithms, system engineering"),
        ("Adobe", "Graphics Software Architect", "c++, object oriented programming, data structures"),
        ("Oracle", "Database Kernel Internals Engineer", "sql, databases, indexing algorithms, transaction maps"),
        ("IBM", "Cognitive Infrastructure SRE", "operating systems, virtualizations, systems automation, infrastructure"),
    ]
    cursor.executemany(
        "INSERT INTO company_roles (company_name, role_name, required_skills) VALUES (?, ?, ?)",
        companies_data
    )

    compiled_inserts = []
    for comp in COMPANIES_LIST:
        for role in ROLES_LIST:
            shuffled = list(QUESTIONS_POOL)
            random.shuffle(shuffled)
            for item in shuffled:
                compiled_inserts.append((comp, role, item[0], item[1], item[2]))

    cursor.executemany(
        "INSERT INTO question_bank (company_name, role_name, subject, source_reference, question) VALUES (?, ?, ?, ?, ?)",
        compiled_inserts
    )

    conn.commit()
    conn.close()

    total_combinations = len(COMPANIES_LIST) * len(ROLES_LIST)
    total_questions = total_combinations * len(QUESTIONS_POOL)
    print("Database seeded successfully - DSA + OS, short 2-3 min questions!")
    print(f"   Companies: {len(COMPANIES_LIST)}")
    print(f"   Roles: {len(ROLES_LIST)}")
    print(f"   Questions per pool: {len(QUESTIONS_POOL)} (10 DSA + 10 OS)")
    print(f"   Total company-role combinations: {total_combinations}")
    print(f"   Total question rows inserted: {total_questions}")


if __name__ == "__main__":
    init_db()