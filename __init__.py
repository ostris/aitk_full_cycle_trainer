# This is an example extension for custom training. It is great for experimenting with new ideas.
from toolkit.extension import Extension


# This is for generic training (LoRA, Dreambooth, FineTuning)
class SDFullCycleTrainerExtension(Extension):
    # uid must be unique, it is how the extension is identified
    uid = "full_cycle_trainer"

    # name is the name of the extension for printing
    name = "Full Cycle Trainer"

    # This is where your process class is loaded
    # keep your imports in here so they don't slow down the rest of the program
    @classmethod
    def get_process(cls):
        # import your process class here so it is only loaded when needed and return it
        from .SDFullCycleTrainer import SDFullCycleTrainer
        return SDFullCycleTrainer

# This is for generic training (LoRA, Dreambooth, FineTuning)
class SDTurboTrainer(Extension):
    # uid must be unique, it is how the extension is identified
    uid = "turbo_trainer"

    # name is the name of the extension for printing
    name = "Turbo Trainer"

    # This is where your process class is loaded
    # keep your imports in here so they don't slow down the rest of the program
    @classmethod
    def get_process(cls):
        # import your process class here so it is only loaded when needed and return it
        from .TurboTrainer import SDTurboTrainer
        return SDTurboTrainer


AI_TOOLKIT_EXTENSIONS = [
    # you can put a list of extensions here
    SDFullCycleTrainerExtension, SDTurboTrainer
]
