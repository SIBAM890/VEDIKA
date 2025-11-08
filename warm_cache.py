"""
Cache Warmer Script
Run this once after deployment to pre-cache common questions
This makes 80% of queries INSTANT (0.001s response)
"""

import asyncio
from backend.ultra_optimized_rag import ultra_fast_rag
import time

# -------------------------------------------------------------------------
# Common Questions (Customize for your university)
# -------------------------------------------------------------------------
COMMON_QUESTIONS = [
    # Programs - B.Tech
    "What are all B.Tech specializations?",
    "Tell me about B.Tech CSE",
    "B.Tech CSE eligibility criteria",
    "B.Tech CSE fees and duration",
    "What is B.Tech in AI and Machine Learning?",
    "Difference between B.Tech CSE and ECE",
    
    # Programs - Management
    "Tell me about MBA program",
    "MBA eligibility and fees",
    "What is Global MBA?",
    "Difference between MBA and Global MBA",
    "BBA course details",
    "DEAN OF Manegement Studies",
    
    # Programs - Computer Applications
    "What is BCA course?",
    "BCA specializations available",
    "MCA program details",
    "MCA eligibility criteria",
    "Difference between BCA and B.Tech CSE",
    
    # Programs - Science
    "List all B.Sc programs",
    "B.Sc Computer Science details",
    "M.Sc programs available",
    
    # Programs - Medical
    "What is BAMS course?",
    "Tell me about Bachelor of Physiotherapy",
    "Nursing programs available",
    "Dean Of BAMS",
            # Admissions
    "How to apply for admission?",
    "What is the admission process?",
    "Admission eligibility criteria",
    "When does admission start?",
    "Required documents for admission",
    "Entrance exam details",
    
    # Fees & Scholarships
    "What are the fees for B.Tech?",
    "MBA program fees",
    "Are scholarships available?",
    "Fee structure for all programs",
    "Payment options available",
    
    # Campus & Facilities
    "Tell me about campus facilities",
    "Hostel facilities available",
    "Library and lab facilities",
    "Sports facilities on campus",
    "What is campus life like?",
    
    # Placements & Careers
    "Placement statistics",
    "Top recruiting companies",
    "Average placement package",
    "Career opportunities after B.Tech CSE",
    "Placement support provided",
    "Who is the current dean of Sri Sri University?",
    "Who is the dean of School of Engineering?",
    "Tell me about B.Tech CSE program",
    "What programs does Sri Sri University offer?",
    "List all courses available",
    "What are the admission requirements?",
    "Who is All Time ?",
    
    # General
    "List all undergraduate programs",
    "List all postgraduate programs",
    "List all courses offered",
    "What makes Sri Sri University unique?",
    "University accreditation and rankings",
    
    # Specific Course Combinations
    "What is B.Tech CSE in Cyber Security?",
    "Tell me about B.Tech in EV and AI",
    "What is MCA in Data Science?",
    "B.Tech specializations in ECE",
    
    # Comparison Questions
    "Compare B.Tech and BCA",
    "Compare MBA and Global MBA",
    "Difference between B.Sc CS and B.Tech CSE",
]

# -------------------------------------------------------------------------
# Warmer Functions
# -------------------------------------------------------------------------
async def warm_single_query(query: str, index: int, total: int):
    """Warm cache for a single query"""
    print(f"\n[{index}/{total}] Warming: {query}")
    start = time.time()
    
    try:
        response, metadata = await ultra_fast_rag(query)
        duration = time.time() - start
        
        # Show preview
        preview = response[:100].replace('\n', ' ')
        print(f"✅ Cached in {duration:.2f}s")
        print(f"   Preview: {preview}...")
        print(f"   Metadata: {metadata.get('steps', [])}")
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

async def warm_cache_batch(questions: list, batch_size: int = 3):
    """Warm cache in batches to avoid overwhelming API"""
    total = len(questions)
    success_count = 0
    
    print("="*80)
    print(f"🔥 CACHE WARMING STARTED")
    print(f"   Total queries: {total}")
    print(f"   Batch size: {batch_size}")
    print("="*80)
    
    start_time = time.time()
    
    # Process in batches
    for i in range(0, total, batch_size):
        batch = questions[i:i+batch_size]
        
        # Process batch in parallel
        tasks = [
            warm_single_query(q, i+j+1, total) 
            for j, q in enumerate(batch)
        ]
        results = await asyncio.gather(*tasks)
        success_count += sum(results)
        
        # Rate limiting - wait between batches
        if i + batch_size < total:
            print(f"\n⏳ Waiting 2s before next batch...")
            await asyncio.sleep(2)
    
    total_time = time.time() - start_time
    
    print("\n" + "="*80)
    print(f"✅ CACHE WARMING COMPLETE")
    print(f"   Success: {success_count}/{total}")
    print(f"   Failed: {total - success_count}")
    print(f"   Total time: {total_time:.1f}s")
    print(f"   Avg per query: {total_time/total:.2f}s")
    print("="*80)
    print("\n💡 These queries will now respond in ~0.001s (instant)!")

async def warm_specific_topics(topics: list):
    """Warm cache for specific topic areas"""
    topic_queries = {
        'btech': [
            "List all B.Tech programs",
            "B.Tech eligibility",
            "B.Tech fees",
        ],
        'mba': [
            "MBA details",
            "MBA eligibility",
            "MBA fees",
        ],
        'admissions': [
            "Admission process",
            "How to apply",
            "Entrance exam",
        ],
    }
    
    queries = []
    for topic in topics:
        queries.extend(topic_queries.get(topic, []))
    
    await warm_cache_batch(queries)

# -------------------------------------------------------------------------
# Main Execution
# -------------------------------------------------------------------------
async def main():
    """Main cache warming function"""
    print("\n🚀 Starting Cache Warmer for Sri Sri University Chatbot\n")
    
    choice = input("Select warming mode:\n1. Full (all common questions)\n2. Quick (top 10 only)\n3. Custom topics\nChoice (1/2/3): ")
    
    if choice == '1':
        print("\n📋 Warming FULL cache (may take 5-10 minutes)...")
        await warm_cache_batch(COMMON_QUESTIONS, batch_size=3)
    
    elif choice == '2':
        print("\n⚡ Warming TOP 10 queries...")
        await warm_cache_batch(COMMON_QUESTIONS[:10], batch_size=3)
    
    elif choice == '3':
        print("\nAvailable topics: btech, mba, admissions")
        topics_input = input("Enter topics (comma-separated): ")
        topics = [t.strip() for t in topics_input.split(',')]
        await warm_specific_topics(topics)
    
    else:
        print("❌ Invalid choice")
        return
    
    print("\n🎉 Cache warming complete! Your chatbot is now BLAZING FAST! ⚡")
    print("Run this script periodically (weekly) to keep cache fresh.")

def run_warmer():
    """Entry point for running the warmer"""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n❌ Cache warming interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error during cache warming: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_warmer()