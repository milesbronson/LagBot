CREATE TABLE IF NOT EXISTS hands (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(64) NOT NULL,
    hand_number INTEGER NOT NULL,
    num_players INTEGER NOT NULL,
    small_blind INTEGER NOT NULL,
    big_blind INTEGER NOT NULL,
    community_cards TEXT[],
    pot INTEGER NOT NULL,
    winner_ids INTEGER[],
    winner_amounts INTEGER[],
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS hand_actions (
    id SERIAL PRIMARY KEY,
    hand_id INTEGER REFERENCES hands(id) ON DELETE CASCADE,
    action_order INTEGER NOT NULL,
    player_id INTEGER NOT NULL,
    player_name VARCHAR(64),
    betting_round VARCHAR(16),
    action_type VARCHAR(16),
    amount INTEGER DEFAULT 0,
    pot_after INTEGER,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS hand_players (
    id SERIAL PRIMARY KEY,
    hand_id INTEGER REFERENCES hands(id) ON DELETE CASCADE,
    player_id INTEGER NOT NULL,
    player_name VARCHAR(64),
    hole_cards TEXT[],
    starting_stack INTEGER,
    ending_stack INTEGER,
    is_human BOOLEAN DEFAULT FALSE,
    position VARCHAR(16)
);

CREATE INDEX IF NOT EXISTS idx_hands_session ON hands(session_id);
CREATE INDEX IF NOT EXISTS idx_hand_actions_hand ON hand_actions(hand_id);
CREATE INDEX IF NOT EXISTS idx_hand_players_hand ON hand_players(hand_id);
