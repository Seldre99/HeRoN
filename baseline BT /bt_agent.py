SUCCESS = 1
FAILURE = 0

# Base Nodes
class Node:
    def tick(self, env):
        raise NotImplementedError


class Selector(Node):
    def __init__(self, children):
        self.children = children

    def tick(self, env):
        for child in self.children:
            if child.tick(env) == SUCCESS:
                return SUCCESS
        return FAILURE


class Sequence(Node):
    def __init__(self, children):
        self.children = children

    def tick(self, env):
        for child in self.children:
            if child.tick(env) == FAILURE:
                return FAILURE
        return SUCCESS


class Condition(Node):
    def __init__(self, fn):
        self.fn = fn

    def tick(self, env):
        return SUCCESS if self.fn(env) else FAILURE


class Action(Node):
    def __init__(self, fn):
        self.fn = fn

    def tick(self, env):
        env.last_action = self.fn(env)
        return SUCCESS

# Action IDs
ACTION_ATTACK   = 0
ACTION_FIRE     = 1
ACTION_THUNDER  = 2
ACTION_BLIZZARD = 3
ACTION_METEOR   = 4
ACTION_CURA     = 5
ACTION_POTION  = 6
ACTION_GRENADE = 7
ACTION_ELIXIR  = 8

# Conditions
def hp_ratio(env):
    p = env.players[0]
    return p.get_hp() / p.maxhp


def hp_low(env, threshold):
    return hp_ratio(env) <= threshold


def mp_at_least(cost):
    def _cond(env):
        return env.players[0].get_mp() >= cost
    return _cond


def mp_too_low_for_cure(env):
    return env.players[0].get_mp() < 32


def has_grenade(env):
    return env.players[0].items[1]["quantity"] > 0


def has_elixir(env):
    return env.players[0].items[2]["quantity"] > 0

# Actions
def do_attack(env):   return ACTION_ATTACK
def do_fire(env):     return ACTION_FIRE
def do_thunder(env):  return ACTION_THUNDER
def do_blizzard(env): return ACTION_BLIZZARD
def do_meteor(env):   return ACTION_METEOR
def do_cura(env):     return ACTION_CURA
def do_grenade(env):  return ACTION_GRENADE
def do_elixir(env):   return ACTION_ELIXIR


def build_bt():
    return Selector([

        # ---- HEAL ----
        Sequence([
            Condition(lambda env: hp_low(env, 0.4)),
            Condition(mp_at_least(32)),
            Action(do_cura)
        ]),

        # ---- EMERGENCY ELIXIR ----
        Sequence([
            Condition(lambda env: hp_low(env, 0.3)),
            Condition(mp_too_low_for_cure),
            Condition(has_elixir),
            Action(do_elixir)
        ]),

        # ---- OFFENSIVE MAGIC (MP RANGE FALLBACK) ----
        Sequence([
            Condition(mp_at_least(40)),
            Action(do_meteor)
        ]),

        Sequence([
            Condition(mp_at_least(35)),
            Action(do_blizzard)
        ]),

        Sequence([
            Condition(mp_at_least(30)),
            Action(do_thunder)
        ]),

        Sequence([
            Condition(mp_at_least(25)),
            Action(do_fire)
        ]),

        # ---- ITEM FALLBACK ----
        Sequence([
            Condition(has_grenade),
            Action(do_grenade)
        ]),

        # ---- FINAL FALLBACK ----
        Action(do_attack)
    ])

class BTAgent:
    def __init__(self):
        self.tree = build_bt()
        self.last_action = None

    def act(self, env):
        self.tree.tick(env)
        return env.last_action
