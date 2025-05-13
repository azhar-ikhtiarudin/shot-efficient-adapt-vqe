from src.pools import QE # Other Options: NoZPauliPool, SingletGSD, CEO, DVG_CEO
from src.molecules import create_h2 # Other Options: create_h3, create_h4, create_beh2

from algorithms.adapt_vqe_original import AdaptVQE
from algorithms.adapt_vqe import AdaptVQE # Comment this line to use the original AdaptVQE


if __name__ == '__main__':    
    r = 0.742
    molecule = create_h2(r)
    pool = QE(molecule)

    adapt_vqe = AdaptVQE(pool=pool,
                        molecule=molecule,
                        max_adapt_iter=100,
                        max_opt_iter=100,
                        grad_threshold=1e-4,
                        vrb=True,
                        optimizer_method='l-bfgs-b',
                        shots_assignment='uniform',
                        k=100,
                        shots_budget=1024,
                        N_experiments=2
                        )

    adapt_vqe.run()
