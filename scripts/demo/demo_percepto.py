#!/usr/bin/env python3
"""
Interactive demo of the PerceptoEmocional agent.
"""
import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from cerebrum_artis.agents import PerceptoEmocional


def main():
    parser = argparse.ArgumentParser(description='PerceptoEmocional Demo')
    parser.add_argument('--image_path', type=str, help='Path to artwork image')
    parser.add_argument('--caption', type=str, default='', help='Caption describing the artwork')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    parser.add_argument('--checkpoint', type=str, default=None, help='Model checkpoint path')
    
    args = parser.parse_args()
    
    # Initialize agent
    print("🎨 Initializing PerceptoEmocional agent...")
    agent = PerceptoEmocional(checkpoint_path=args.checkpoint)
    print("✓ Agent ready!\n")
    
    if args.interactive:
        print("Interactive Mode - Enter 'quit' to exit")
        print("-" * 50)
        
        while True:
            image_path = input("\nImage path: ").strip()
            if image_path.lower() in ['quit', 'exit', 'q']:
                break
                
            caption = input("Caption (optional): ").strip()
            
            try:
                result = agent.analyze_painting(image_path, caption if caption else None)
                
                print("\n" + "=" * 50)
                print(f"Emotion: {result['emotion']}")
                print(f"Confidence: {result['confidence']:.2%}")
                print(f"\nExplanation:")
                print(result['explanation'])
                print("=" * 50)
                
            except Exception as e:
                print(f"Error: {e}")
    
    elif args.image_path:
        # Single image analysis
        result = agent.analyze_painting(args.image_path, args.caption if args.caption else None)
        
        print("=" * 50)
        print(f"Emotion: {result['emotion']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"\nExplanation:")
        print(result['explanation'])
        
        if 'probabilities' in result:
            print(f"\nTop 3 emotions:")
            probs = result['probabilities']
            top_3 = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]
            for emotion, prob in top_3:
                print(f"  {emotion}: {prob:.2%}")
        print("=" * 50)
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
