import os
from story_system import StorySystem

"""
Before submitting the assignment, describe here in a few sentences what you would have built next if you spent 2 more hours on this project:

If I had 2 more hours, I would have:
1. Added story categorization (adventure, friendship, educational) with tailored generation strategies
2. Implemented user feedback loops allowing iterative story refinement based on user preferences
3. Created a story library system to save and retrieve favorite stories
4. Added more sophisticated prompt engineering techniques like few-shot examples for different story types
5. Implemented A/B testing framework to continuously improve prompt effectiveness
"""

def main():
    """
    Main interface for the Hippocratic AI Bedtime Story Generator
    
    Features:
    - Multi-agent story generation with quality control
    - Age-appropriate content for children 5-10 years old
    - Automatic evaluation and refinement
    - Safety-first approach with content validation
    """
    
    print("🌙 Welcome to the Hippocratic AI Bedtime Story Generator! 🌙")
    print("=" * 60)
    print("I create safe, age-appropriate bedtime stories for children ages 5-10.")
    print("Each story goes through quality evaluation and refinement.")
    print("=" * 60)
    
    try:
        # Initialize the story system
        story_system = StorySystem()
        
        # Test communication first
        print("\n🔍 Verifying system communication...")
        if not story_system.verify_communication(no_print=True):
            print("❌ System communication test failed. Please check the setup.")
            return
        
        print("\n✅ System ready! Let's create your bedtime story.\n")
        
        # Get user input
        user_input = input("What kind of bedtime story would you like to hear?\n\n")
        
        if not user_input.strip():
            print("Please provide a story idea! For example:")
            print("- A brave little mouse who learns to share")
            print("- A magical garden where vegetables come to life") 
            print("- Two best friends who go on a gentle adventure")
            return
        
        print(f"\n🎭 Creating your story: '{user_input}'")
        print("This may take a moment as we generate, evaluate, and refine your story...\n")
        
        # Generate the story using the multi-agent system
        result = story_system.create_story(user_input, max_iterations=1, no_print=True)
        
        # Display the final result
        story_system.print_story_result(result)
        
        # Optional: Ask for user feedback
        print("\nWould you like to request any changes to the story? (y/n)")
        feedback_choice = input().lower().strip()
        
        if feedback_choice == 'y':
            feedback = input("What would you like me to change or improve? ")
            if feedback.strip():
                print(f"\n🔄 Refining story based on your feedback: '{feedback}'")
                
                # Use the refiner to make user-requested changes
                improved_story = story_system.refiner.improve(result['story'], feedback)
                
                print("\n" + "="*60)
                print("🌟 REFINED STORY 🌟")
                print("="*60)
                print(improved_story)
                print("="*60)
        
        print("\n🌙 Sweet dreams! Thank you for using the Bedtime Story Generator! 🌙")
        
    except KeyboardInterrupt:
        print("\n\n👋 Story generation cancelled. Sweet dreams!")
    
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        print("Please make sure your OPENAI_API_KEY environment variable is set.")
        print("You can set it by running: export OPENAI_API_KEY='your-api-key-here'")


if __name__ == "__main__":
    main()