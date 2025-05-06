import subprocess
import time
import random
import pyautogui  

# List of diverse query components
prefixes = [
    "How does", "Why do", "What is the future of", "Beginner’s guide to",
    "Step-by-step explanation of", "The impact of", "History of", "Latest research on",
    "Top 5 reasons for", "Can you explain", "Best way to learn about", "Is it true that"
]

topics = [
    "quantum computing", "deep space exploration", "climate change",
    "AI vs human intelligence", "the evolution of smartphones",
    "dark web and cybersecurity", "history of video games", "how nuclear energy works",
    "mysteries of black holes", "latest AI breakthroughs", "why do humans dream",
    "best renewable energy sources", "impact of social media", "history of programming languages",
    "how electric cars work", "hidden secrets of the ocean", "mind control technology",
    "the science of happiness"
]

suffixes = [
    "explained", "for beginners", "pros and cons", "future trends",
    "in detail", "step-by-step", "impact on society", "latest updates",
    "case study", "vs traditional methods", "detailed analysis", "should you invest in this?"
]

# Function to generate a random search query
def generate_query():
    query = f"{random.choice(prefixes)} {random.choice(topics)} {random.choice(suffixes)}"
    
    # Occasionally add slight variations to make it seem more natural
    if random.random() < 0.3:  # 30% chance to add a variation
        query += f" {random.choice(['in 2025', 'with real-world examples', 'based on research'])}"
    
    return query

# Function to open Microsoft Edge and perform a search
def search_with_edge():
    query = generate_query()

    # Open Edge manually (macOS version)
    subprocess.run(["open", "-a", "Microsoft Edge"])
    
    # Wait for Edge to open (adjust delay if needed)
    time.sleep(random.uniform(3, 5))  
    
    # Type the search query (simulating human typing)
    pyautogui.write(query, interval=random.uniform(0.08, 0.2))  # Random typing speed
    pyautogui.press("enter")  # Press Enter to search
    
    print(f"Searching: {query} on Edge")

# Define session properties
max_searches = random.randint(30, 100)  # Random number of searches per session

for _ in range(max_searches):
    search_with_edge()
    
    # Randomized delay before the next search
    sleep_time = (2)  # Wait between 10 and 45 seconds
    time.sleep(sleep_time)

print("Session completed. Stopping automated searches.")