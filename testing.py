if __name__ == "__main__":
    from policyengine_us import Microsimulation

    sim = Microsimulation()
    print(sim.tax_benefit_system.variables.keys())
