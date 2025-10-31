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

        ;Fat agent predicates (any agent with enough food to take/extend lead)
        (fat_agent ?x) ;agent has enough food to take lead if they score
        (fat_agent_gt3 ?x) ;agent has enough food to extend lead by >3
        (fat_agent_gt10 ?x) ;agent has enough food to extend lead by >10

        ;Capsule predicate (global - current agent is closest)
        (closest_to_capsule) ;current agent is closest to a capsule

        ;Enemy distance predicates (global, not per-enemy)
        (enemy_breathing_distance) ;at least one enemy within breathing distance
        (enemy_close_distance) ;at least one enemy within close distance

        ;Team coordination predicates
        (ally_chasing ?e - enemy) ;ally is currently chasing this enemy
        (chasing ?e - enemy ?a - team) ;agent is chasing this enemy

        ;Cooperative predicates
        (eat_enemy ?a - ally)
        (go_home ?a - ally)
        (go_enemy_land ?a - ally)
        (eat_capsule ?a - ally)
        (eat_food ?a - ally)
        (defend ?a - ally)

    )

    ;define actions here

    ;Attack - cross into enemy territory to get food
    (:action attack
        :parameters (?a - current_agent)
        :precondition (and
            (not (is_pacman ?a))
        )
        :effect (and
            (is_pacman ?a)
        )
    )

    ;Eat food - collect food while in enemy territory
    (:action eat_food
        :parameters (?a - current_agent)
        :precondition (and
            (is_pacman ?a)
        )
        :effect (and
            (fat_agent ?a)
            (fat_agent_gt3 ?a)
            (fat_agent_gt10 ?a)
        )
    )

    ;Eat capsule - consume power capsule
    (:action eat_capsule
        :parameters (?a - current_agent ?e1 - enemy ?e2 - enemy)
        :precondition (and
            (closest_to_capsule)
        )
        :effect (and
            (is_scared ?e1)
            (is_scared ?e2)
        )
    )

    ;Escape - return home (stop being pacman)
    (:action escape
        :parameters (?a - current_agent)
        :precondition (and
            (is_pacman ?a)
        )
        :effect (and
            (not (is_pacman ?a))
        )
    )

    ;Defend - default defensive behavior (patrol, guard food)
    (:action defend
        :parameters (?a - current_agent)
        :precondition (and
            (not (is_pacman ?a))
        )
        :effect (and
            (defend_foods)
        )
    )

    ;Chase enemy - actively pursue a specific enemy
    (:action chase_enemy
        :parameters (?a - current_agent ?e - enemy)
        :precondition (and
            (not (is_pacman ?a))
            (not (is_scared ?a))
        )
        :effect (and
            (not (is_pacman ?e))
        )
    )
)
