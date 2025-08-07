if __name__ == "__main__":
    from policyengine_us import Microsimulation

    sim = Microsimulation()
    print(sim.calculate("ucgid_str"))
