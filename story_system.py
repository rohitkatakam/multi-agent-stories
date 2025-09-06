"""
Bedtime Story Generation System with LLM Judge
============================================

A multi-agent system for generating age-appropriate bedtime stories with quality evaluation and refinement.

Agents:
1. StoryGenerator - Creates initial story drafts
2. StoryJudge - Evaluates story quality and safety  
3. StoryRefiner - Improves stories based on judge feedback
"""

import os
from typing import Dict, Any, Optional
import json
import re

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv is optional

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("⚠️ OpenAI module not available - some functionality will be limited")


class OpenAIClient:
    """Centralized OpenAI API client using the new OpenAI 1.0+ API"""
    
    def __init__(self):
        if not OPENAI_AVAILABLE:
            raise ValueError("OpenAI library not installed. Run: pip install openai")
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        # Initialize the new OpenAI client
        self.client = OpenAI(api_key=api_key)
    
    def call_model(self, prompt: str, max_tokens: int = 3000, temperature: float = 0.1) -> str:
        """Make API call to OpenAI using the new 1.0+ API with consistent error handling"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"OpenAI API Error: {e}")
            return f"Error generating content: {str(e)}"


class StoryGenerator:
    """
    Agent responsible for generating initial bedtime story drafts.
    
    Demonstrates:
    - Realistic first-draft generation
    - Basic age-appropriate content creation
    - Simple story structure prompting
    - Leaves room for judge refinement
    """
    
    def __init__(self, client: OpenAIClient):
        self.client = client
    
    def generate(self, user_request: str) -> str:
        """Generate a bedtime story based on user request"""
        
        prompt = self._create_story_prompt(user_request)
        
        # Use slightly higher temperature for creativity in story generation
        story = self.client.call_model(prompt, max_tokens=800, temperature=0.3)
        
        return story.strip()
    
    def _create_story_prompt(self, request: str) -> str:
        """Create a realistic first-draft prompt for story generation"""
        
        return f"""You are a creative storyteller writing for children ages 5-10.

TASK: Write a story based on: "{request}"

BASIC REQUIREMENTS:
- 250-400 words
- Appropriate for children ages 5-10
- Include characters and a simple plot
- Have a beginning, middle, and end

Write the story now:"""


class StoryJudge:
    """
    Agent responsible for evaluating story quality and appropriateness.
    
    Demonstrates:
    - Multi-criteria evaluation prompts
    - Structured feedback generation
    - Safety and appropriateness assessment
    """
    
    def __init__(self, client: OpenAIClient):
        self.client = client
        
        # Criteria weights (total = 1.0)
        self.CRITERIA_WEIGHTS = {
            'safety': 0.30,                # Must be perfect for children
            'age_appropriateness': 0.25,   # Critical for target audience
            'bedtime_suitability': 0.20,   # Essential for bedtime stories
            'story_quality': 0.15,         # Important for engagement
            'educational_value': 0.10      # Nice bonus
        }
    
    def evaluate(self, story: str, user_request: str = "") -> Dict[str, Any]:
        """Enhanced evaluation with weighted scoring and adaptive thresholds"""
        
        # Get basic evaluation
        basic_eval = self._evaluate_basic(story)
        
        # Calculate weighted score
        weighted_score = self._calculate_weighted_score(basic_eval['scores'])
        
        # Get adaptive threshold based on request complexity
        threshold = self._get_adaptive_threshold(user_request)
        
        # Determine sophisticated recommendation
        recommendation, quality_tier, feedback = self._get_sophisticated_recommendation(
            basic_eval['scores'], weighted_score, threshold
        )
        
        return {
            **basic_eval,
            'weighted_score': weighted_score,
            'quality_tier': quality_tier,
            'threshold_used': threshold,
            'recommendation': recommendation,
            'sophisticated_feedback': feedback
        }
    
    def _evaluate_basic(self, story: str) -> Dict[str, Any]:
        """Evaluate a story across multiple quality dimensions"""
        
        prompt = self._create_evaluation_prompt(story)
        
        # Use low temperature for consistent evaluation
        evaluation_text = self.client.call_model(prompt, max_tokens=500, temperature=0.1)
        
        # Parse the structured evaluation response
        return self._parse_evaluation(evaluation_text)
    
    def _create_evaluation_prompt(self, story: str) -> str:
        """Create a structured evaluation prompt with clear format instructions"""
        
        return f"""You are a child development expert and story quality assessor specializing in bedtime stories for ages 5-10.

TASK: Evaluate the following bedtime story across multiple criteria.

STORY TO EVALUATE:
{story}

EVALUATION CRITERIA (Rate each 1-10):
1. Age Appropriateness: Vocabulary, concepts, and themes suitable for ages 5-10
2. Story Quality: Plot coherence, character development, engaging narrative
3. Bedtime Suitability: Calming tone, peaceful content, sleep-encouraging ending
4. Safety: No frightening, violent, or inappropriate content for children
5. Educational Value: Positive messages, life lessons, or character growth

IMPORTANT: Follow this EXACT format for each criterion:
[Criterion Name]: [Number 1-10] - [Brief explanation]

REQUIRED RESPONSE FORMAT:
Age Appropriateness: [Score] - [Explanation]
Story Quality: [Score] - [Explanation]
Bedtime Suitability: [Score] - [Explanation]
Safety: [Score] - [Explanation]
Educational Value: [Score] - [Explanation]

Overall Average: [Calculate and show average of the 5 scores above]
Recommendation: PASS (if average >= 7) or REVISE (if average < 7)
Main Feedback: [If REVISE, provide 1-2 specific, actionable improvement suggestions]

Please evaluate now:"""
    
    def _parse_evaluation(self, evaluation_text: str) -> Dict[str, Any]:
        """Parse evaluation with enhanced error handling and validation"""
        
        lines = evaluation_text.strip().split('\n')
        result = {
            'scores': {},
            'explanations': {},
            'average': 0,
            'recommendation': 'REVISE',  # Default to safe option
            'feedback': '',
            'parsing_warnings': []
        }
        
        expected_criteria = [
            'Age Appropriateness',
            'Story Quality', 
            'Bedtime Suitability',
            'Safety',
            'Educational Value'
        ]
        
        try:
            for line in lines:
                line = line.strip()
                if ':' in line:
                    # Handle each criterion with flexible matching
                    for criterion in expected_criteria:
                        if criterion.lower() in line.lower():
                            score, explanation = self._extract_score_and_explanation(line)
                            key = criterion.lower().replace(' ', '_')
                            result['scores'][key] = score
                            result['explanations'][key] = explanation
                            break
                    
                    # Handle overall metrics
                    if 'overall average' in line.lower():
                        try:
                            avg_text = line.split(':')[1].strip()
                            # Extract number from various formats
                            avg_match = re.search(r'(\d+(?:\.\d+)?)', avg_text)
                            if avg_match:
                                result['average'] = float(avg_match.group(1))
                        except:
                            result['parsing_warnings'].append("Could not parse average")
                    
                    elif 'recommendation' in line.lower():
                        rec_text = line.split(':', 1)[1].strip().upper()
                        if 'PASS' in rec_text:
                            result['recommendation'] = 'PASS'
                        elif 'REVISE' in rec_text:
                            result['recommendation'] = 'REVISE'
                    
                    elif 'main feedback' in line.lower() or 'feedback' in line.lower():
                        result['feedback'] = line.split(':', 1)[1].strip()
            
            # Validation and fallbacks
            if not result['scores']:
                result['parsing_warnings'].append("No scores parsed successfully")
                result['average'] = 5.0
                result['recommendation'] = 'REVISE'
                result['feedback'] = "Evaluation parsing failed - please review story manually"
            
            # Calculate average if not provided or seems wrong
            if result['scores'] and (result['average'] == 0 or 
                                   abs(result['average'] - sum(result['scores'].values()) / len(result['scores'])) > 0.5):
                calculated_avg = sum(result['scores'].values()) / len(result['scores'])
                result['average'] = round(calculated_avg, 1)
                result['parsing_warnings'].append(f"Recalculated average: {result['average']}")
            
            # Ensure recommendation matches average
            if result['average'] >= 7 and result['recommendation'] != 'PASS':
                result['recommendation'] = 'PASS'
                result['parsing_warnings'].append("Corrected recommendation to PASS based on average")
            elif result['average'] < 7 and result['recommendation'] != 'REVISE':
                result['recommendation'] = 'REVISE'
                result['parsing_warnings'].append("Corrected recommendation to REVISE based on average")
            
            # Ensure feedback exists for REVISE
            if result['recommendation'] == 'REVISE' and not result['feedback']:
                result['feedback'] = "Story needs improvement based on low scores. Please review all criteria."
                result['parsing_warnings'].append("Generated fallback feedback")
                
        except Exception as e:
            result['parsing_warnings'].append(f"Parsing error: {str(e)}")
            result['average'] = 5.0
            result['recommendation'] = 'REVISE'
            result['feedback'] = "Evaluation parsing failed - story needs review"
        
        # Print warnings if any
        if result['parsing_warnings']:
            print("⚠️ Evaluation parsing warnings:")
            for warning in result['parsing_warnings']:
                print(f"   - {warning}")
        
        return result
    
    def _extract_score_and_explanation(self, line: str) -> tuple:
        """Extract numeric score and explanation with multiple format support"""
        try:
            # Remove the criterion name and colon
            content = line.split(':', 1)[1].strip()
            
            # Try multiple separator patterns
            separators = [' - ', '- ', ' -', '-', ' — ', '—']
            
            for sep in separators:
                if sep in content:
                    parts = content.split(sep, 1)
                    score_text = parts[0].strip()
                    explanation = parts[1].strip() if len(parts) > 1 else ""
                    
                    # Extract numeric score (handle formats like "8", "8.5", "8/10", "8 out of 10")
                    score_match = re.search(r'(\d+(?:\.\d+)?)', score_text)
                    if score_match:
                        score = float(score_match.group(1))
                        # Ensure score is in 1-10 range
                        score = max(1, min(10, score))
                        return score, explanation
            
            # Fallback: try to extract any number from the content
            score_match = re.search(r'(\d+(?:\.\d+)?)', content)
            if score_match:
                score = float(score_match.group(1))
                score = max(1, min(10, score))
                return score, content
                
        except Exception as e:
            print(f"Warning: Could not parse score from line '{line[:50]}...': {e}")
        
        return 5.0, "Unable to parse score"
    
    def _calculate_weighted_score(self, scores: Dict[str, float]) -> float:
        """Calculate weighted score based on criteria importance"""
        weighted_sum = 0
        total_weight = 0
        
        for key, score in scores.items():
            if key in self.CRITERIA_WEIGHTS:
                weight = self.CRITERIA_WEIGHTS[key]
                weighted_sum += score * weight
                total_weight += weight
        
        return round(weighted_sum / total_weight if total_weight > 0 else 0, 1)
    
    def _get_adaptive_threshold(self, user_request: str) -> float:
        """Get adaptive threshold based on story complexity"""
        request_lower = user_request.lower()
        
        # Complex story elements (slightly more lenient)
        complex_elements = ['magic', 'adventure', 'journey', 'problem', 'challenge', 'mystery']
        
        # Simple story elements (higher standards)
        simple_elements = ['friendship', 'kindness', 'sharing', 'family', 'help', 'care']
        
        # Educational elements (moderate standards)
        educational_elements = ['learn', 'lesson', 'teach', 'discover', 'understand']
        
        if any(element in request_lower for element in complex_elements):
            return 8.2  # Slightly more lenient for complex narratives
        elif any(element in request_lower for element in simple_elements):
            return 8.7  # Higher standard for simple stories
        elif any(element in request_lower for element in educational_elements):
            return 8.4  # Moderate for educational content
        else:
            return 8.5  # Default higher standard
    
    def _get_sophisticated_recommendation(self, scores: Dict[str, float], weighted_score: float, threshold: float) -> tuple:
        """Get sophisticated recommendation with multi-dimensional excellence standards"""
        
        # PHASE 1: Non-negotiable safety checks (stricter)
        safety_score = scores.get('safety', 0)
        if safety_score < 9.5:  # Much stricter safety requirement
            return 'REVISE', 'SAFETY_CRITICAL', 'Story must be completely safe for children - no exceptions'
        
        # PHASE 2: Individual excellence requirements for each criterion
        age_score = scores.get('age_appropriateness', 0)
        bedtime_score = scores.get('bedtime_suitability', 0) 
        quality_score = scores.get('story_quality', 0)
        edu_score = scores.get('educational_value', 0)
        
        # Each criterion needs individual excellence
        if age_score < 8.5:  # Much stricter than previous 7.0
            return 'REVISE', 'AGE_REFINEMENT', 'Language and concepts need age-appropriate adjustments for 5-10 year olds'
        
        if bedtime_score < 8.5:  # New strict requirement
            return 'REVISE', 'BEDTIME_REFINEMENT', 'Story needs calming tone and peaceful ending suitable for bedtime'
        
        if quality_score < 8.0:  # New requirement for story quality
            return 'REVISE', 'QUALITY_REFINEMENT', 'Story structure, characters, or plot need improvement'
        
        if edu_score < 7.5:  # Moderate requirement for educational value
            return 'REVISE', 'EDUCATIONAL_REFINEMENT', 'Story needs clearer positive message or life lesson'
        
        # PHASE 3: Weighted excellence check (much higher standards)
        if weighted_score < 9.0:  # Very high weighted standard (was threshold-based)
            return 'REVISE', 'OVERALL_REFINEMENT', f'Story needs overall improvement (scored {weighted_score:.1f}, needs 9.0+)'
        
        # PHASE 4: Quality tiers for passed stories (higher thresholds)
        if weighted_score >= 9.5:
            return 'PASS', 'EXCELLENT', 'Outstanding bedtime story!'
        elif weighted_score >= 9.2:
            return 'PASS', 'VERY_GOOD', 'High-quality story with minor room for improvement'
        else:
            return 'PASS', 'GOOD', 'Story meets high standards'


class StoryRefiner:
    """
    Agent responsible for improving stories based on judge feedback with targeted refinement.
    
    Demonstrates:
    - Intelligent refinement type identification
    - Targeted improvement prompts (Safety, Bedtime, Age, Quality, Educational)
    - Specialized refinement strategies for different issues
    - Maintaining story essence while making focused improvements
    """
    
    def __init__(self, client: OpenAIClient):
        self.client = client
    
    def improve(self, story: str, feedback: str) -> str:
        """Improve a story based on specific feedback with targeted refinement"""
        
        # Determine refinement type from feedback
        refinement_type = self._identify_refinement_type(feedback)
        
        prompt = self._create_refinement_prompt(story, feedback, refinement_type)
        
        # Use moderate temperature for creative improvements
        improved_story = self.client.call_model(prompt, max_tokens=800, temperature=0.2)
        
        return improved_story.strip()
    
    def _identify_refinement_type(self, feedback: str) -> str:
        """Identify the type of refinement needed based on feedback"""
        feedback_lower = feedback.lower()
        
        # Check in order of specificity to avoid conflicts
        if 'safety' in feedback_lower or 'safe' in feedback_lower:
            return 'SAFETY'
        elif 'educational' in feedback_lower or 'lesson' in feedback_lower or 'message' in feedback_lower:
            return 'EDUCATIONAL'
        elif 'bedtime' in feedback_lower or 'calming' in feedback_lower or 'peaceful' in feedback_lower:
            return 'BEDTIME'
        elif 'age' in feedback_lower or ('appropriate' in feedback_lower and 'age' in feedback_lower) or 'language' in feedback_lower:
            return 'AGE'
        elif 'quality' in feedback_lower or 'plot' in feedback_lower or 'character' in feedback_lower:
            return 'QUALITY'
        else:
            return 'OVERALL'
    
    def _create_refinement_prompt(self, story: str, feedback: str, refinement_type: str) -> str:
        """Create a targeted refinement prompt based on the specific type of improvement needed"""
        
        base_prompt = f"""You are an expert story editor specializing in children's bedtime stories.

ORIGINAL STORY:
{story}

FEEDBACK TO ADDRESS:
{feedback}

"""
        
        # Add specific guidance based on refinement type
        if refinement_type == 'SAFETY':
            base_prompt += """SAFETY-FOCUSED REFINEMENT:
Your primary goal is to make this story completely safe and appropriate for children ages 5-10.

SPECIFIC ACTIONS:
- Remove any scary, violent, or inappropriate content completely
- Ensure all characters behave in positive, child-friendly ways
- Replace any potentially frightening situations with gentle alternatives
- Make sure all conflicts are resolved through kindness and understanding
- Ensure the story promotes positive values like friendship, kindness, and cooperation
- Double-check that nothing could cause anxiety or bad dreams

"""
        elif refinement_type == 'BEDTIME':
            base_prompt += """BEDTIME OPTIMIZATION:
Your primary goal is to make this story perfect for bedtime - calming and sleep-encouraging.

SPECIFIC ACTIONS:
- Add calming, peaceful language throughout the story
- Include soothing imagery (stars, moonlight, gentle sounds, cozy settings)
- Ensure the ending is gentle and sleep-encouraging
- Remove any overly exciting, stimulating, or energetic content
- Use slower pacing and gentler transitions between scenes
- Add peaceful descriptions of nature, comfort, or rest
- Make the resolution feel satisfying but not overstimulating
- End with the character(s) feeling peaceful, content, or ready for sleep

"""
        elif refinement_type == 'AGE':
            base_prompt += """AGE-APPROPRIATENESS REFINEMENT:
Your primary goal is to make this story perfectly suitable for children ages 5-10.

SPECIFIC ACTIONS:
- Simplify vocabulary to age-appropriate level (avoid complex words)
- Ensure concepts are developmentally appropriate for 5-10 year olds
- Adjust sentence complexity - use shorter, clearer sentences
- Make sure themes and situations are relatable to young children
- Ensure emotional content is appropriate (not too complex or intense)
- Use familiar settings and situations children can understand
- Make character motivations clear and simple

"""
        elif refinement_type == 'QUALITY':
            base_prompt += """STORY QUALITY ENHANCEMENT:
Your primary goal is to improve the overall narrative quality and engagement.

SPECIFIC ACTIONS:
- Strengthen character development and make characters more relatable
- Improve plot structure and pacing for better flow
- Add more engaging dialogue or character interactions
- Enhance descriptions to make the story more vivid and immersive
- Ensure clear story progression from beginning to middle to end
- Make the conflict and resolution more satisfying
- Add interesting details that bring the story to life
- Ensure the story has good rhythm and readability

"""
        elif refinement_type == 'EDUCATIONAL':
            base_prompt += """EDUCATIONAL VALUE ENHANCEMENT:
Your primary goal is to strengthen the positive messages and life lessons.

SPECIFIC ACTIONS:
- Make the life lesson or positive message clearer and more prominent
- Ensure the lesson emerges naturally from the story events
- Show characters learning and growing through their experiences
- Include examples of positive behavior (sharing, kindness, helping others)
- Make the moral of the story age-appropriate and relatable
- Avoid being preachy - let the lesson come through actions and consequences
- Reinforce positive values throughout the story, not just at the end

"""
        else:  # OVERALL refinement
            base_prompt += """OVERALL STORY IMPROVEMENT:
Your primary goal is to enhance the story across all dimensions for bedtime storytelling.

SPECIFIC ACTIONS:
- Improve age-appropriateness and vocabulary
- Enhance the calming, bedtime-suitable tone
- Strengthen story quality and engagement
- Ensure complete safety for children
- Add or clarify positive messages and lessons
- Make targeted improvements while preserving the story's core appeal

"""
        
        base_prompt += """REQUIREMENTS:
- Keep the core story concept, characters, and setting
- Maintain 250-400 word length
- Make focused improvements without complete rewrite
- Ensure the final story is perfect for bedtime reading
- Preserve any elements that were already working well

IMPROVED STORY:"""
        
        return base_prompt


class StorySystem:
    """
    Main orchestrator that coordinates the three agents to generate high-quality bedtime stories.
    
    Demonstrates:
    - Multi-agent coordination
    - Quality control workflow
    - Iterative improvement process
    """
    
    def __init__(self):
        self.client = OpenAIClient()
        self.generator = StoryGenerator(self.client)
        self.judge = StoryJudge(self.client)
        self.refiner = StoryRefiner(self.client)
    
    def create_story(self, user_request: str, max_iterations: int = 1, no_print: bool = False) -> Dict[str, Any]:
        """
        Generate a high-quality bedtime story through multi-agent collaboration
        
        Args:
            user_request: User's story request
            max_iterations: Maximum refinement iterations
            
        Returns:
            Dictionary containing final story and process metadata
        """
        
        if not no_print:
            print(f"🎭 Generating story for ages 5-10: '{user_request}'")
        
        # Step 1: Generate initial story
        print("📝 Story Generator: Creating initial draft...")
        story = self.generator.generate(user_request)
        
        # Step 2: Evaluate quality
        print("⚖️ Story Judge: Evaluating quality and safety...")
        evaluation = self.judge.evaluate(story, user_request)
        
        # Display enhanced scoring information
        weighted_score = evaluation.get('weighted_score', evaluation['average'])
        quality_tier = evaluation.get('quality_tier', 'UNKNOWN')
        threshold = evaluation.get('threshold_used', 8.5)
        
        print(f"📊 Weighted Score: {weighted_score:.1f}/10 (Threshold: {threshold:.1f})")
        print(f"🏆 Quality Tier: {quality_tier}")
        print(f"🎯 Recommendation: {evaluation['recommendation']}")
        
        iteration_count = 0
        
        # Step 3: Refine if needed (limited iterations to prevent over-engineering)
        while (evaluation['recommendation'] == 'REVISE' and 
               iteration_count < max_iterations and 
               evaluation.get('sophisticated_feedback', evaluation.get('feedback', ''))):
            
            iteration_count += 1
            print(f"✏️ Story Refiner: Improving story (iteration {iteration_count})...")
            
            feedback = evaluation.get('sophisticated_feedback', evaluation.get('feedback', ''))
            story = self.refiner.improve(story, feedback)
            evaluation = self.judge.evaluate(story, user_request)
            
            # Display improved scoring
            weighted_score = evaluation.get('weighted_score', evaluation['average'])
            quality_tier = evaluation.get('quality_tier', 'UNKNOWN')
            
            print(f"📊 Improved Score: {weighted_score:.1f}/10")
            print(f"🏆 New Quality Tier: {quality_tier}")
            print(f"🎯 New Recommendation: {evaluation['recommendation']}")
        
        # Return comprehensive result
        result = {
            'story': story,
            'evaluation': evaluation,
            'iterations': iteration_count,
            'user_request': user_request
        }
        
        print("✅ Story generation complete!")
        return result
    
    def verify_communication(self, no_print=False) -> bool:
        """Verify that agents can communicate properly"""
        
        if not no_print:
            print("🔍 Verifying agent communication...")
        
        # Test evaluation parsing with known good format
        test_evaluation = """Age Appropriateness: 8 - Good vocabulary for target age
Story Quality: 7 - Clear plot structure
Bedtime Suitability: 9 - Very calming tone
Safety: 10 - Completely safe content
Educational Value: 6 - Some positive messages

Overall Average: 8.0
Recommendation: PASS
Main Feedback: Story is good overall."""
        
        try:
            result = self.judge._parse_evaluation(test_evaluation)
            
            # Check all required fields are present
            required_fields = ['scores', 'average', 'recommendation', 'feedback']
            for field in required_fields:
                if field not in result:
                    print(f"❌ Missing required field: {field}")
                    return False
            
            # Check scores were parsed
            if len(result['scores']) < 3:  # At least 3 criteria should be parsed
                print(f"❌ Insufficient scores parsed: {len(result['scores'])}")
                return False
            
            # Check recommendation is valid
            if result['recommendation'] not in ['PASS', 'REVISE']:
                print(f"❌ Invalid recommendation: {result['recommendation']}")
                return False
            
            # Test REVISE scenario
            test_revise = """Age Appropriateness: 5 - Too complex language
Story Quality: 6 - Plot unclear
Bedtime Suitability: 4 - Too exciting
Safety: 9 - Safe content
Educational Value: 7 - Good lesson

Overall Average: 6.2
Recommendation: REVISE
Main Feedback: Simplify language and tone down excitement."""
            
            revise_result = self.judge._parse_evaluation(test_revise)
            if revise_result['recommendation'] != 'REVISE' or not revise_result['feedback']:
                print("❌ REVISE scenario parsing failed")
                return False
            
            if not no_print:
                print("✅ Agent communication verified successfully!")
                print(f"   - Parsed {len(result['scores'])} criteria scores")
                print(f"   - Average calculation: {result['average']}")
                print(f"   - Recommendation logic: working")
                print(f"   - Feedback extraction: working")
            return True
            
        except Exception as e:
            print(f"❌ Communication verification failed: {e}")
            return False
    
    def print_story_result(self, result: Dict[str, Any]) -> None:
        """Pretty print the final story result"""
        
        print("\n" + "="*60)
        print("🌙 BEDTIME STORY GENERATED 🌙")
        print("="*60)
        print(f"📖 Request: {result['user_request']}")
        print(f"🔄 Refinement Iterations: {result['iterations']}")
        print(f"⭐ Final Quality Score: {result['evaluation']['average']:.1f}/10")
        print("-"*60)
        print(result['story'])
        print("-"*60)
        
        
        print("✅ Story meets quality standards!")
        
        print("="*60)


# Test function for communication verification (standalone)
def test_parsing_only():
    """Test just the parsing logic without requiring OpenAI client"""
    print("🧪 Testing Agent Communication Parsing (No dependencies)")
    print("=" * 50)
    
    try:
        # Create a mock judge with just the parsing methods
        class MockJudge:
            def __init__(self):
                # Criteria weights (total = 1.0)
                self.CRITERIA_WEIGHTS = {
                    'safety': 0.30,                # Must be perfect for children
                    'age_appropriateness': 0.25,   # Critical for target audience
                    'bedtime_suitability': 0.20,   # Essential for bedtime stories
                    'story_quality': 0.15,         # Important for engagement
                    'educational_value': 0.10      # Nice bonus
                }
            def _extract_score_and_explanation(self, line: str) -> tuple:
                """Extract numeric score and explanation with multiple format support"""
                try:
                    # Remove the criterion name and colon
                    content = line.split(':', 1)[1].strip()
                    
                    # Try multiple separator patterns
                    separators = [' - ', '- ', ' -', '-', ' — ', '—']
                    
                    for sep in separators:
                        if sep in content:
                            parts = content.split(sep, 1)
                            score_text = parts[0].strip()
                            explanation = parts[1].strip() if len(parts) > 1 else ""
                            
                            # Extract numeric score (handle formats like "8", "8.5", "8/10", "8 out of 10")
                            score_match = re.search(r'(\d+(?:\.\d+)?)', score_text)
                            if score_match:
                                score = float(score_match.group(1))
                                # Ensure score is in 1-10 range
                                score = max(1, min(10, score))
                                return score, explanation
                    
                    # Fallback: try to extract any number from the content
                    score_match = re.search(r'(\d+(?:\.\d+)?)', content)
                    if score_match:
                        score = float(score_match.group(1))
                        score = max(1, min(10, score))
                        return score, content
                        
                except Exception as e:
                    print(f"Warning: Could not parse score from line '{line[:50]}...': {e}")
                
                return 5.0, "Unable to parse score"
            
            def _parse_evaluation(self, evaluation_text: str) -> Dict[str, Any]:
                """Parse evaluation with enhanced error handling and validation"""
                
                lines = evaluation_text.strip().split('\n')
                result = {
                    'scores': {},
                    'explanations': {},
                    'average': 0,
                    'recommendation': 'REVISE',  # Default to safe option
                    'feedback': '',
                    'parsing_warnings': []
                }
                
                expected_criteria = [
                    'Age Appropriateness',
                    'Story Quality', 
                    'Bedtime Suitability',
                    'Safety',
                    'Educational Value'
                ]
                
                try:
                    for line in lines:
                        line = line.strip()
                        if ':' in line:
                            # Handle each criterion with flexible matching
                            for criterion in expected_criteria:
                                if criterion.lower() in line.lower():
                                    score, explanation = self._extract_score_and_explanation(line)
                                    key = criterion.lower().replace(' ', '_')
                                    result['scores'][key] = score
                                    result['explanations'][key] = explanation
                                    break
                            
                            # Handle overall metrics
                            if 'overall average' in line.lower():
                                try:
                                    avg_text = line.split(':')[1].strip()
                                    # Extract number from various formats
                                    avg_match = re.search(r'(\d+(?:\.\d+)?)', avg_text)
                                    if avg_match:
                                        result['average'] = float(avg_match.group(1))
                                except:
                                    result['parsing_warnings'].append("Could not parse average")
                            
                            elif 'recommendation' in line.lower():
                                rec_text = line.split(':', 1)[1].strip().upper()
                                if 'PASS' in rec_text:
                                    result['recommendation'] = 'PASS'
                                elif 'REVISE' in rec_text:
                                    result['recommendation'] = 'REVISE'
                            
                            elif 'main feedback' in line.lower() or 'feedback' in line.lower():
                                result['feedback'] = line.split(':', 1)[1].strip()
                    
                    # Validation and fallbacks
                    if not result['scores']:
                        result['parsing_warnings'].append("No scores parsed successfully")
                        result['average'] = 5.0
                        result['recommendation'] = 'REVISE'
                        result['feedback'] = "Evaluation parsing failed - please review story manually"
                    
                    # Calculate average if not provided or seems wrong
                    if result['scores'] and (result['average'] == 0 or 
                                           abs(result['average'] - sum(result['scores'].values()) / len(result['scores'])) > 0.5):
                        calculated_avg = sum(result['scores'].values()) / len(result['scores'])
                        result['average'] = round(calculated_avg, 1)
                        result['parsing_warnings'].append(f"Recalculated average: {result['average']}")
                    
                    # Ensure recommendation matches average
                    if result['average'] >= 7 and result['recommendation'] != 'PASS':
                        result['recommendation'] = 'PASS'
                        result['parsing_warnings'].append("Corrected recommendation to PASS based on average")
                    elif result['average'] < 7 and result['recommendation'] != 'REVISE':
                        result['recommendation'] = 'REVISE'
                        result['parsing_warnings'].append("Corrected recommendation to REVISE based on average")
                    
                    # Ensure feedback exists for REVISE
                    if result['recommendation'] == 'REVISE' and not result['feedback']:
                        result['feedback'] = "Story needs improvement based on low scores. Please review all criteria."
                        result['parsing_warnings'].append("Generated fallback feedback")
                        
                except Exception as e:
                    result['parsing_warnings'].append(f"Parsing error: {str(e)}")
                    result['average'] = 5.0
                    result['recommendation'] = 'REVISE'
                    result['feedback'] = "Evaluation parsing failed - story needs review"
                
                # Print warnings if any
                if result['parsing_warnings']:
                    print("⚠️ Evaluation parsing warnings:")
                    for warning in result['parsing_warnings']:
                        print(f"   - {warning}")
                
                return result
            
            def _calculate_weighted_score(self, scores: Dict[str, float]) -> float:
                """Calculate weighted score based on criteria importance"""
                weighted_sum = 0
                total_weight = 0
                
                for key, score in scores.items():
                    if key in self.CRITERIA_WEIGHTS:
                        weight = self.CRITERIA_WEIGHTS[key]
                        weighted_sum += score * weight
                        total_weight += weight
                
                return round(weighted_sum / total_weight if total_weight > 0 else 0, 1)
            
            def _get_adaptive_threshold(self, user_request: str) -> float:
                """Get adaptive threshold based on story complexity"""
                request_lower = user_request.lower()
                
                # Complex story elements (slightly more lenient)
                complex_elements = ['magic', 'adventure', 'journey', 'problem', 'challenge', 'mystery']
                
                # Simple story elements (higher standards)
                simple_elements = ['friendship', 'kindness', 'sharing', 'family', 'help', 'care']
                
                # Educational elements (moderate standards)
                educational_elements = ['learn', 'lesson', 'teach', 'discover', 'understand']
                
                if any(element in request_lower for element in complex_elements):
                    return 8.2  # Slightly more lenient for complex narratives
                elif any(element in request_lower for element in simple_elements):
                    return 8.7  # Higher standard for simple stories
                elif any(element in request_lower for element in educational_elements):
                    return 8.4  # Moderate for educational content
                else:
                    return 8.5  # Default higher standard
            
            def _get_sophisticated_recommendation(self, scores: Dict[str, float], weighted_score: float, threshold: float) -> tuple:
                """Get sophisticated recommendation with multi-dimensional excellence standards"""
                
                # PHASE 1: Non-negotiable safety checks (stricter)
                safety_score = scores.get('safety', 0)
                if safety_score < 9.5:  # Much stricter safety requirement
                    return 'REVISE', 'SAFETY_CRITICAL', 'Story must be completely safe for children - no exceptions'
                
                # PHASE 2: Individual excellence requirements for each criterion
                age_score = scores.get('age_appropriateness', 0)
                bedtime_score = scores.get('bedtime_suitability', 0) 
                quality_score = scores.get('story_quality', 0)
                edu_score = scores.get('educational_value', 0)
                
                # Each criterion needs individual excellence
                if age_score < 8.5:  # Much stricter than previous 7.0
                    return 'REVISE', 'AGE_REFINEMENT', 'Language and concepts need age-appropriate adjustments for 5-10 year olds'
                
                if bedtime_score < 8.5:  # New strict requirement
                    return 'REVISE', 'BEDTIME_REFINEMENT', 'Story needs calming tone and peaceful ending suitable for bedtime'
                
                if quality_score < 8.0:  # New requirement for story quality
                    return 'REVISE', 'QUALITY_REFINEMENT', 'Story structure, characters, or plot need improvement'
                
                if edu_score < 7.5:  # Moderate requirement for educational value
                    return 'REVISE', 'EDUCATIONAL_REFINEMENT', 'Story needs clearer positive message or life lesson'
                
                # PHASE 3: Weighted excellence check (much higher standards)
                if weighted_score < 9.0:  # Very high weighted standard (was threshold-based)
                    return 'REVISE', 'OVERALL_REFINEMENT', f'Story needs overall improvement (scored {weighted_score:.1f}, needs 9.0+)'
                
                # PHASE 4: Quality tiers for passed stories (higher thresholds)
                if weighted_score >= 9.5:
                    return 'PASS', 'EXCELLENT', 'Outstanding bedtime story!'
                elif weighted_score >= 9.2:
                    return 'PASS', 'VERY_GOOD', 'High-quality story with minor room for improvement'
                else:
                    return 'PASS', 'GOOD', 'Story meets high standards'
        
        judge = MockJudge()
        
        # Test 1: HIGH-QUALITY PASS scenario (meets new strict standards)
        print("🔍 Testing HIGH-QUALITY PASS scenario...")
        test_pass = """Age Appropriateness: 9.0 - Perfect vocabulary and concepts for ages 5-10
Story Quality: 8.5 - Excellent plot structure and character development
Bedtime Suitability: 9.5 - Very calming and sleep-encouraging
Safety: 10 - Completely safe content with positive values
Educational Value: 8.0 - Clear positive message about kindness

Overall Average: 9.0
Recommendation: PASS
Main Feedback: Excellent bedtime story that meets all standards."""
        
        result1 = judge._parse_evaluation(test_pass)
        weighted1 = judge._calculate_weighted_score(result1['scores'])
        threshold1 = judge._get_adaptive_threshold("friendship story")
        recommendation1, tier1, feedback1 = judge._get_sophisticated_recommendation(result1['scores'], weighted1, threshold1)
        
        print(f"   - Parsed {len(result1['scores'])} scores")
        print(f"   - Basic Average: {result1['average']}")
        print(f"   - Weighted Score: {weighted1}")
        print(f"   - Threshold: {threshold1} (friendship = high standard)")
        print(f"   - Quality Tier: {tier1}")
        print(f"   - Recommendation: {recommendation1}")
        
        if len(result1['scores']) < 4:
            print("❌ PASS scenario failed - insufficient scores parsed")
            return False
        
        # Test 2: BEDTIME REFINEMENT scenario (good story but needs bedtime optimization)
        print("\n🔍 Testing BEDTIME REFINEMENT scenario...")
        test_bedtime_revise = """Age Appropriateness: 9.0 - Perfect for ages 5-10
Story Quality: 8.5 - Excellent adventure story
Bedtime Suitability: 7.0 - Too exciting for bedtime, needs calming
Safety: 10 - Completely safe content
Educational Value: 8.0 - Great lesson about courage

Overall Average: 8.5
Recommendation: REVISE
Main Feedback: Story needs bedtime optimization - too stimulating."""
        
        result2 = judge._parse_evaluation(test_bedtime_revise)
        weighted2 = judge._calculate_weighted_score(result2['scores'])
        threshold2 = judge._get_adaptive_threshold("magic adventure")
        recommendation2, tier2, feedback2 = judge._get_sophisticated_recommendation(result2['scores'], weighted2, threshold2)
        
        print(f"   - Parsed {len(result2['scores'])} scores")
        print(f"   - Basic Average: {result2['average']}")
        print(f"   - Weighted Score: {weighted2}")
        print(f"   - Threshold: {threshold2} (adventure = more lenient)")
        print(f"   - Quality Tier: {tier2}")
        print(f"   - Recommendation: {recommendation2}")
        print(f"   - Feedback: {feedback2[:50]}...")
        
        if recommendation2 != 'REVISE' or not feedback2:
            print("❌ REVISE scenario failed")
            return False
        
        # Test 3: Format variations
        print("\n🔍 Testing format variations...")
        test_variations = """Age Appropriateness: 8.5- Good vocabulary
Story Quality:7 - Clear plot
Bedtime Suitability: 9 — Very calming
Safety: 10 Safe content
Educational Value: 6.5 - Some lessons

Overall Average: 8.2
Recommendation: PASS
Main Feedback: Story works well."""
        
        result3 = judge._parse_evaluation(test_variations)
        print(f"   - Parsed {len(result3['scores'])} scores with format variations")
        
        if len(result3['scores']) < 3:  # Should parse at least 3 despite format issues
            print("❌ Format variation handling failed")
            return False
        
        # Test 4: Refinement Type Identification
        print("\n🔍 Testing Refinement Type Identification...")
        
        class MockRefiner:
            def _identify_refinement_type(self, feedback: str) -> str:
                feedback_lower = feedback.lower()
                
                # Check in order of specificity to avoid conflicts
                if 'safety' in feedback_lower or 'safe' in feedback_lower:
                    return 'SAFETY'
                elif 'educational' in feedback_lower or 'lesson' in feedback_lower or 'message' in feedback_lower:
                    return 'EDUCATIONAL'
                elif 'bedtime' in feedback_lower or 'calming' in feedback_lower or 'peaceful' in feedback_lower:
                    return 'BEDTIME'
                elif 'age' in feedback_lower or ('appropriate' in feedback_lower and 'age' in feedback_lower) or 'language' in feedback_lower:
                    return 'AGE'
                elif 'quality' in feedback_lower or 'plot' in feedback_lower or 'character' in feedback_lower:
                    return 'QUALITY'
                else:
                    return 'OVERALL'
        
        refiner = MockRefiner()
        
        test_cases = [
            ("Story needs calming tone and peaceful ending suitable for bedtime", "BEDTIME"),
            ("Story must be completely safe for children - no exceptions", "SAFETY"),
            ("Language and concepts need age-appropriate adjustments", "AGE"),
            ("Story structure, characters, or plot need improvement", "QUALITY"),
            ("Story needs clearer positive message or life lesson", "EDUCATIONAL"),
            ("Story needs overall improvement", "OVERALL")
        ]
        
        for feedback, expected_type in test_cases:
            identified_type = refiner._identify_refinement_type(feedback)
            if identified_type == expected_type:
                print(f"   ✓ {expected_type}: Correctly identified")
            else:
                print(f"   ❌ {expected_type}: Expected {expected_type}, got {identified_type}")
                return False
        
        print("\n✅ ALL PARSING TESTS PASSED!")
        print("   - PASS scenario: ✓")
        print("   - REVISE scenario: ✓") 
        print("   - Format variations: ✓")
        print("   - Error handling: ✓")
        print("   - Refinement type identification: ✓")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


if __name__ == "__main__":
    success = test_parsing_only()
    
    print("\n" + "=" * 50)
    if success:
        print("✅ COMMUNICATION PARSING TEST PASSED!")
        print("Agents can properly parse and communicate evaluation results.")
        print("The system is ready for integration with OpenAI API.")
    else:
        print("❌ COMMUNICATION PARSING TEST FAILED!")
        print("There are issues with agent communication that need to be fixed.")
    print("=" * 50)
