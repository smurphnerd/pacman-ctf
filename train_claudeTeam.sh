#!/bin/bash
# Training script for claudeTeam Q-learning weights

echo "=========================================="
echo "ClaudeTeam Q-Learning Training Script"
echo "=========================================="
echo ""

# Default values
GAMES=100
OPPONENT="berkeleyTeam"
LAYOUT="defaultCapture"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--games)
            GAMES="$2"
            shift 2
            ;;
        -o|--opponent)
            OPPONENT="$2"
            shift 2
            ;;
        -l|--layout)
            LAYOUT="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: ./train_claudeTeam.sh [options]"
            echo ""
            echo "Options:"
            echo "  -n, --games N       Number of training games (default: 100)"
            echo "  -o, --opponent TEAM Opponent team (default: berkeleyTeam)"
            echo "  -l, --layout LAYOUT Map layout (default: defaultCapture)"
            echo "  -h, --help          Show this help message"
            echo ""
            echo "Examples:"
            echo "  ./train_claudeTeam.sh -n 500"
            echo "  ./train_claudeTeam.sh -n 200 -o baselineTeam"
            echo "  ./train_claudeTeam.sh -n 300 -l mediumCapture"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

echo "Training Configuration:"
echo "  Games: $GAMES"
echo "  Opponent: $OPPONENT"
echo "  Layout: $LAYOUT"
echo ""

# Check if weights file exists
if [ -f "QLWeightsClaudeTeam.pkl" ]; then
    echo "Found existing weights file: QLWeightsClaudeTeam.pkl"
    echo "Training will continue from previous weights."
else
    echo "No existing weights found. Starting fresh training."
fi
echo ""

# Run training
echo "Starting training..."
echo "=========================================="
python capture.py -r claudeTeam -b $OPPONENT -x $GAMES -q -l $LAYOUT

echo ""
echo "=========================================="
echo "Training complete!"
echo ""

# Check if weights were saved
if [ -f "QLWeightsClaudeTeam.pkl" ]; then
    echo "✓ Weights saved successfully to QLWeightsClaudeTeam.pkl"
    echo ""
    echo "To continue training, run this script again."
    echo "To test the trained agent, run:"
    echo "  python capture.py -r claudeTeam -b $OPPONENT -n 10"
else
    echo "⚠ Warning: Weights file not found. Training may have failed."
fi
echo "=========================================="
