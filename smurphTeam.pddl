;Header and description

(define (domain pacman_bool)

    ;remove requirements that are not needed
    (:requirements :strips :typing)
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
        (winning)   ; is the team score more than enemy team
        (is_scared ?x) ;is enemy, current agent, or the ally in panic (due to capsule eaten by other side)
        (is_pacman ?x) ; if an agent is pacman
        (can_win_with_backpack ?x) ; the score in current backpack is enough to take the lead

        (enemy_around ?e - enemy ?a - team) ;enemy ?e is within 4 grid distance with agent ?a

        ;Predicates for virtual state to set goal states
        (defend_foods) ;The environment do not collect state for this predicates, this is a virtual effect state for action patrol


        ;Advanced predicates
        ;These predicates are currently not used and consider the state of other agent
        (enemy_passed ?e - enemy ?a - current_agent) ; enemy got past the current agent and is now eating food

        ;Cooperative predicates
        (more_enemies_around ?a - ally) ; ally has more enemies around them

    )

    ;define actions here

    (:action invade
        :parameters (?a1 - current_agent)
        :precondition (and
            (not (is_pacman ?a1))
        )
        :effect (and
            (is_pacman ?a1)
            (not (is_scared ?a1))
        )
    )

    (:action eat_food_until_winning
        :parameters (?a1 - current_agent ?a2 - ally ?e1 - enemy1 ?e2 - enemy2)
        :precondition (and
            (is_pacman ?a1)
            (not (winning))
            (not (can_win_with_backpack ?a1))
            (or(
                (not (enemy_around ?e1 ?a1))
                (is_scared ?e1)
            ))
            (or(
                (not (enemy_around ?e2 ?a1))
                (is_scared ?e2)
            ))
        )
        :effect (and
            (can_win_with_backpack ?a1)
        )
    )

    (:action escape_enemy
        :parameters (?a - current_agent ?e - enemy1)
        :precondition (and
            (is_pacman ?a)
            (not (is_scared ?e))
            (enemy_around ?e ?a)
        )
        :effect (and
            (not (enemy_around ?e ?a))
        )
    )

    (:action go_home
        :parameters (?a - current_agent)
        :precondition (and
            (is_pacman ?a)
        )
        :effect (and
            (not (is_pacman ?a))
        )
    )

    (:action take_lead
        :parameters (?a - current_agent)
        :precondition (and
            (can_win_with_backpack ?a)
        )
        :effect (and
            (winning)
            (not (can_win_with_backpack ?a))
            (not (is_pacman ?a))
        )
    )

    (:action defend_territory
        :parameters (?a1 - current_agent ?a2 - ally ?e1 - enemy ?e2 - enemy)
        :precondition (and
            (not (is_pacman ?a1))
            (not (is_scared ?a1))
            (not (enemy_passed ?e1 ?a1))
            (not (enemy_passed ?e2 ?a1))
        )
        :effect (and
            (defend_foods)
            (not (is_pacman ?e1))
            (not (is_pacman ?e2))
        )
    )

    (:action prevent_escape
        :parameters (?a - current_agent ?e - enemy)
        :precondition (and
            (enemy_passed ?e ?a)
            (is_pacman ?e)
            (not (is_scared ?a))
        )
        :effect (and
            (not (is_pacman ?e))
        )
    )
)
