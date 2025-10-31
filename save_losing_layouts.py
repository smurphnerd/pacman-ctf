#!/usr/bin/env python
"""
Wrapper script to run capture.py and save layouts where your team (red) loses.
Usage: python save_losing_layouts.py -r myTeam.py -b staffTeam.py -l RANDOM -n 40 -q
"""

import sys
import os
import capture
from capture import readCommand, runGames

def save_layout(layout, game_num, score):
    """Save a layout to a .lay file"""
    os.makedirs('losing_layouts', exist_ok=True)
    filename = f'losing_layouts/loss_game{game_num}_score{score}.lay'

    with open(filename, 'w') as f:
        f.write(str(layout))

    print(f'Saved losing layout to: {filename}')

def main():
    # Parse command line arguments
    options = readCommand(sys.argv[1:])

    # Extract parameters
    layouts = options['layouts']
    agents = options['agents']
    display = options['display']
    length = options['length']
    numGames = options['numGames']
    record = options['record']
    numTraining = options['numTraining']
    redTeamName = options['redTeamName']
    blueTeamName = options['blueTeamName']
    muteAgents = options.get('muteAgents', False)
    catchExceptions = options['catchExceptions']
    results_out = options['results_out']

    # Run the games
    games = runGames(
        layouts, agents, display, length, numGames, record,
        numTraining, redTeamName, blueTeamName, muteAgents,
        catchExceptions, results_out
    )

    # Save layouts where red team (your team with -r flag) lost
    print('\n' + '='*60)
    print('Checking for losses to save layouts...')
    print('='*60)

    losses_saved = 0
    for i, game in enumerate(games):
        score = game.state.data.score
        layout = layouts[i]

        # Red team loses when score is negative
        if score < 0:
            save_layout(layout, i+1, score)
            losses_saved += 1

    print(f'\nTotal layouts saved: {losses_saved}')
    print(f'Layouts saved in: losing_layouts/ directory')

    # Also save the overall score like the original
    capture.save_score(games[0])

if __name__ == '__main__':
    main()
