from repgroup import RepGroup, RepGroupElement

class LieGroup(RepGroup):
    def __init__(self, represent, depresent, identity):
        
        super().__init__(represent, depresent, identity)

