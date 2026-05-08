from skill import SelfEvolvingSkill, QueryContext, FeedbackType


def demonstrate_self_evolving_skill():
    print("=" * 60)
    print("Self-Evolving Skill Demonstration")
    print("=" * 60)
    
    skill = SelfEvolvingSkill()
    
    print("\n1. Adding Knowledge to the System")
    print("-" * 40)
    
    knowledge_items = [
        {
            'content': 'Python is a high-level programming language known for its readability.',
            'tags': ['python', 'programming', 'language']
        },
        {
            'content': 'Machine learning is a subset of AI that enables systems to learn from data.',
            'tags': ['ml', 'ai', 'data']
        },
        {
            'content': 'Self-evolving systems can improve their performance over time.',
            'tags': ['self-improvement', 'systems', 'optimization']
        },
        {
            'content': 'Feedback loops are essential for continuous improvement.',
            'tags': ['feedback', 'optimization', 'systems']
        }
    ]
    
    for item in knowledge_items:
        entry_id = skill.add_knowledge(item['content'], item['tags'])
        print(f"Added: {item['content'][:50]}... (ID: {entry_id})")
    
    print("\n2. Querying the System")
    print("-" * 40)
    
    queries = [
        "What is Python?",
        "How do machines learn?",
        "What makes systems self-improving?"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        result = skill.query(query)
        print(f"Answer: {result['answer'][:80]}...")
        print(f"Confidence: {result['confidence']:.2f}")
        print(f"Requires Human Input: {result.get('requires_human_input', False)}")
    
    print("\n3. Providing Feedback")
    print("-" * 40)
    
    query = "What is Python?"
    result = skill.query(query)
    entry_id = result['sources'][0]
    
    print(f"Giving positive feedback for: {query}")
    skill.provide_feedback(query, entry_id, FeedbackType.POSITIVE)
    
    print(f"Knowledge updated with success count!")
    
    print("\n4. Learning from Interactions")
    print("-" * 40)
    
    skill.learn_from_interaction(
        query="Tell me about Python",
        expected_response="Python is a high-level programming language with simple syntax.",
        actual_response="Python is a programming language.",
        success=True
    )
    print("System learned from successful interaction")
    
    skill.learn_from_interaction(
        query="Complex AI topic",
        expected_response="AI involves neural networks and deep learning architectures.",
        actual_response="AI is artificial intelligence.",
        success=False
    )
    print("System learned from failed interaction")
    
    print("\n5. System Status")
    print("-" * 40)
    
    status = skill.get_system_status()
    print(f"Total Entries: {status['total_entries']}")
    print(f"System Ready: {status['system_ready']}")
    print(f"Performance Metrics:")
    for key, value in status['performance_metrics'].items():
        if isinstance(value, float):
            print(f"  - {key}: {value:.3f}")
        else:
            print(f"  - {key}: {value}")
    
    print("\n6. Evolution Parameters")
    print("-" * 40)
    
    recommendations = skill.evolution_core.get_recommendations()
    print("Current Parameters:")
    for param, value in recommendations['current_params'].items():
        print(f"  - {param}: {value:.3f}")
    
    if recommendations['suggestions']:
        print("\nSuggestions:")
        for suggestion in recommendations['suggestions']:
            print(f"  - {suggestion}")
    
    print("\n7. Context-Aware Adaptation")
    print("-" * 40)
    
    contexts = ['technical', 'creative', 'detailed']
    for ctx in contexts:
        context_obj = QueryContext(domain=ctx)
        params = skill.evolution_core.adapt_to_context(ctx)
        print(f"\nContext: {ctx}")
        print(f"  Relevance Weight: {params['relevance_weight']:.2f}")
        print(f"  Diversity Weight: {params['diversity_weight']:.2f}")
    
    print("\n8. Export Learning")
    print("-" * 40)
    
    learning = skill.export_learning()
    print(f"Exported {len(learning['knowledge_entries'])} knowledge entries")
    print(f"Learned Parameters: {len(learning['learned_parameters'])} parameters")
    
    print("\n" + "=" * 60)
    print("Demonstration Complete!")
    print("=" * 60)


if __name__ == "__main__":
    demonstrate_self_evolving_skill()
