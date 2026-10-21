"""Name -> factory for every agent on the shared engine."""
from . import cs224r as C, figment as F, figgiebot as B, figgieauto as A

AGENTS = {
    # CS224R baselines (Yang 2026)
    "Random": C.RandomAgent, "FairValue": C.FairValue, "Conservative": C.Conservative,
    "MarketMaker": C.MarketMaker, "Bayesian": C.Bayesian,
    # mneha05/figment
    "fig.Fundamental": F.Fundamental, "fig.BayesianMM": F.BayesianMM, "fig.AdaptiveMM": F.AdaptiveMM,
    "fig.Momentum": F.Momentum, "fig.Random": F.FigRandom,
    # revilo-25/Figgie-bot
    "fb.PotMM": B.PotMM, "fb.HandMM": B.HandMM, "fb.FlowMM": B.FlowMM, "fb.FixedSpread": B.FixedSpread,
    "fb.Momentum": B.MomentumBot, "fb.Random": B.FBRandom, "fb.ConstVal": B.ConstVal,
    "fb.PotValShuffled": B.PotValShuffled,
    # 0xDub/figgie-auto
    "fa.TiltInventory": A.TiltInventory, "fa.Spread": A.Spread, "fa.Seller": A.Seller, "fa.Noisy": A.Noisy,
    "fa.PickOff": A.PickOff, "fa.TheHoarder": A.TheHoarder, "fa.PrayingMantis": A.PrayingMantis,
}


def make(name, seed=None):
    cls = AGENTS[name]
    try:
        return cls(seed=seed)
    except TypeError:
        return cls()
