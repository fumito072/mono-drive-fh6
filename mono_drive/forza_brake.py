"""Coast during fleeting predictions; brake immediately for close hazards."""
class BrakeDemandFilter:
    def __init__(self,confirm_seconds=.20):
        self.confirm_seconds=confirm_seconds
        self.requests={}

    def step(self,key,brake,emergency,immediate,now):
        if brake<=.01:
            self.requests.pop(key,None)
            return 0.,False
        previous=self.requests.get(key)
        start=now if previous is None or now-previous[1]>.35 else previous[0]
        self.requests[key]=(start,now)
        # Keep the dictionary bounded when a detector retires old track IDs.
        self.requests={k:v for k,v in self.requests.items() if now-v[1]<1.}
        if immediate or now-start>=self.confirm_seconds:
            return brake,emergency
        return 0.,False


def hazard_pedals(command, speed, raw_brake, confirmed_brake, urgent):
    """Do not turn every single-frame prediction into an accelerator release."""
    steer,gas,brake=command
    if urgent or confirmed_brake>.01:
        return (steer,0.,max(brake,confirmed_brake) if speed>1. else 0.)
    if raw_brake>.01:
        # Close contacts bypass confirmation above. A distant tentative box
        # tempers acceleration while it is checked, rather than dropping RT.
        return (steer,min(gas,.18),brake)
    return command
