;Header and description
;Enhanced PDDL domain for Pacman Capture the Flag
;Implements sophisticated strategic planning with team coordination

(define (domain pacman_ctf_advanced)

    ;remove requirements that are not needed
    (:requirements :strips :typing :negative-preconditions)
    (:types
        enemy team - object
        enemy1 enemy2 - enemy
        ally current_agent - team
    )

    ; un-comment following line if constants are needed
    ;(:constants )
    ;(:types food)
    (:predicates

        ;Basic predicates
        (enemy_around ?e - enemy ?a - team) ;enemy ?e is within 4 grid distance with agent ?a
        (is_pacman ?x) ; if an agent is pacman
        (food_in_backpack ?a - team)  ; have food in backpack
        (food_available) ; still have food on enemy land

        ;Predicates for virtual state to set goal states
        (defend_foods) ;Virtual effect state for patrol action
        (team_coordinated) ;Virtual state indicating team is well coordinated


        ;Advanced predicates
        (enemy_long_distance ?e - enemy ?a - current_agent) ; noisy distance return longer than 25
        (enemy_medium_distance ?e - enemy ?a - current_agent) ; noisy distance return longer than 15
        (enemy_short_distance ?e - enemy ?a - current_agent) ; noisy distance return shorter than 15

        (3_food_in_backpack ?a - team) ; more than 3 food in backpack
        (5_food_in_backpack ?a - team)  ; more than 5 food in backpack
        (10_food_in_backpack ?a - team)    ; more than 10 food in backpack
        (20_food_in_backpack ?a - team)    ; more than 20 food in backpack

        (near_food ?a - current_agent)  ; a food within 4 grid distance
        (near_capsule ?a - current_agent)   ;a capsule within 4 grid distance
        (capsule_available) ; capsule available on map
        (winning)   ; is the team score more than enemy team
        (winning_gt3) ; is the team score 3 more than enemy team
        (winning_gt5)    ; is the team score 5 more than enemy team
        (winning_gt10)  ; is the team score 10 more than enemy team
        (winning_gt20)  ; is the team score 20 more than enemy team
        (near_ally) ; is ally near 4 grid distance
        (is_scared ?x) ;is enemy, current agent, or the ally in panic (due to capsule eaten by other side)

        ;New strategic predicates
        (enemy_carrying_food ?e - enemy) ;enemy has food in backpack
        (can_catch_enemy ?e - enemy ?a - current_agent) ;agent can safely catch enemy
        (safe_to_attack ?a - current_agent) ;safe to invade enemy territory
        (at_chokepoint ?a - current_agent) ;agent is at strategic chokepoint
        (enemy_nearby_food ?e - enemy) ;enemy is threatening our food
        (should_retreat ?a - current_agent) ;agent should return home (ghosts nearby)
        (enough_food_to_lead ?a - current_agent) ;agent has enough food to take/extend lead
        (low_time_remaining) ;less than 300 moves remaining
        (very_low_time_remaining) ;less than 100 moves remaining
        (ally_defending ?a - ally) ;ally is in defensive mode
        (ally_attacking ?a - ally) ;ally is in offensive mode
        (more_enemies_around_ally) ;ally has more enemies nearby than current agent
        (food_cluster_nearby ?a - current_agent) ;multiple food pellets nearby


        ;Cooperative predicates
        (eat_enemy ?a - ally)
        (go_home ?a - ally)
        (go_enemy_land ?a - ally)
        (eat_capsule ?a - ally)
        (eat_food ?a - ally)
        (defend ?a - ally)

    )

    ;define actions here

    ;Offensive action - invade enemy territory to collect food
    (:action attack
        :parameters (?a - current_agent ?e1 - enemy1 ?e2 - enemy2)
        :precondition (and
            (food_available)
            (safe_to_attack ?a)
            (not (is_scared ?a))
        )
        :effect (and
            (is_pacman ?a)
            (food_in_backpack ?a)
            (not (food_available))
        )
    )

    ;Aggressive offensive - attack even when enemies present
    (:action aggressive_attack
        :parameters (?a - current_agent ?e1 - enemy1 ?e2 - enemy2)
        :precondition (and
            (food_available)
            (is_scared ?e1)
            (is_scared ?e2)
            (not (winning_gt10))
        )
        :effect (and
            (is_pacman ?a)
            (food_in_backpack ?a)
            (not (food_available))
        )
    )

    ;Opportunistic attack - go for food clusters
    (:action collect_food_cluster
        :parameters (?a - current_agent)
        :precondition (and
            (food_cluster_nearby ?a)
            (is_pacman ?a)
            (not (should_retreat ?a))
        )
        :effect (and
            (food_in_backpack ?a)
        )
    )

    ;Get power capsule for offensive advantage
    (:action eat_capsule
        :parameters (?a - current_agent ?e1 - enemy1 ?e2 - enemy2)
        :precondition (and
            (near_capsule ?a)
            (capsule_available)
        )
        :effect (and
            (is_scared ?e1)
            (is_scared ?e2)
            (not (capsule_available))
        )
    )

    ;Return home with food
    (:action go_home_with_food
        :parameters (?a - current_agent)
        :precondition (and
            (is_pacman ?a)
            (food_in_backpack ?a)
        )
        :effect (and
            (not (is_pacman ?a))
            (not (food_in_backpack ?a))
        )
    )

    ;Return home when threatened (retreat)
    (:action go_home_retreat
        :parameters (?a - current_agent)
        :precondition (and
            (is_pacman ?a)
            (should_retreat ?a)
        )
        :effect (and
            (not (is_pacman ?a))
        )
    )

    ;Emergency retreat with significant food
    (:action emergency_retreat
        :parameters (?a - current_agent ?e - enemy)
        :precondition (and
            (is_pacman ?a)
            (5_food_in_backpack ?a)
            (enemy_around ?e ?a)
            (not (is_scared ?e))
        )
        :effect (and
            (not (is_pacman ?a))
        )
    )

    ;Defensive action - eliminate invaders
    (:action defence
        :parameters (?a - current_agent ?e - enemy)
        :precondition (and
            (is_pacman ?e)
            (not (is_pacman ?a))
            (enemy_around ?e ?a)
        )
        :effect (and
            (not (is_pacman ?e))
        )
    )

    ;Chase invader - priority chase for invaders carrying food
    (:action chase_invader
        :parameters (?a - current_agent ?e - enemy)
        :precondition (and
            (is_pacman ?e)
            (enemy_carrying_food ?e)
            (not (is_pacman ?a))
            (not (is_scared ?a))
        )
        :effect (and
            (not (is_pacman ?e))
            (not (enemy_carrying_food ?e))
        )
    )

    ;Chase any invader (even without food)
    (:action chase_any_invader
        :parameters (?a - current_agent ?e - enemy)
        :precondition (and
            (is_pacman ?e)
            (not (is_pacman ?a))
            (not (is_scared ?a))
            (enemy_around ?e ?a)
        )
        :effect (and
            (not (is_pacman ?e))
        )
    )

    ;Patrol defensive territory when winning
    (:action patrol
        :parameters (?a - current_agent ?e1 - enemy1 ?e2 - enemy2)
        :precondition (and
            (not (is_pacman ?a))
            (not (is_pacman ?e1))
            (not (is_pacman ?e2))
            (winning_gt5)
        )
        :effect (and
            (defend_foods)
        )
    )

    ;Guard strategic chokepoint
    (:action guard_chokepoint
        :parameters (?a - current_agent)
        :precondition (and
            (at_chokepoint ?a)
            (not (is_pacman ?a))
            (winning)
        )
        :effect (and
            (defend_foods)
        )
    )

    ;Coordinated defense - stay near threatened food
    (:action defend_vulnerable_food
        :parameters (?a - current_agent ?e - enemy)
        :precondition (and
            (enemy_nearby_food ?e)
            (not (is_pacman ?a))
            (not (ally_defending ?a))
        )
        :effect (and
            (defend_foods)
        )
    )

    ;Support ally under pressure
    (:action support_ally
        :parameters (?a - current_agent ?ally - ally)
        :precondition (and
            (more_enemies_around_ally)
            (near_ally)
            (not (is_pacman ?a))
        )
        :effect (and
            (team_coordinated)
        )
    )

    ;Endgame strategy - secure win with time running out
    (:action secure_win
        :parameters (?a - current_agent ?e1 - enemy1 ?e2 - enemy2)
        :precondition (and
            (very_low_time_remaining)
            (winning)
            (not (is_pacman ?e1))
            (not (is_pacman ?e2))
        )
        :effect (and
            (defend_foods)
        )
    )

    ;Endgame desperation - force score when losing
    (:action desperate_attack
        :parameters (?a - current_agent)
        :precondition (and
            (very_low_time_remaining)
            (not (winning))
            (food_available)
        )
        :effect (and
            (is_pacman ?a)
            (food_in_backpack ?a)
            (not (food_available))
        )
    )

)