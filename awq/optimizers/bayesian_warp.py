verbose = False
# Number of pairwise comparisons performed before checking posterior mean
every_n_comps = 3
# Total number of checking the maximum posterior mean
n_check_post_mean = 5
n_outcome_model_initialization_points = 8
n_reps = 1
within_session_results = []
exp_candidate_results = []

for i in range(n_reps):
    print(f"Run {i}")
    # Experimentation stage: initial exploration batch
    torch.manual_seed(i)
    np.random.seed(i)
    X, Y = generate_random_exp_data(problem, n_outcome_model_initialization_points)
    outcome_model = fit_outcome_model(X, Y, problem.bounds)

    # Preference exploration stage: initialize the preference model with comparsions
    # between pairs of outcomes estimated using random design points
    init_train_Y, init_train_comps = generate_random_pref_data(outcome_model, n=1)

    # Perform preference exploration using either Random-f or EUBO-zeta
    for pe_strategy in ["EUBO-zeta", "Random-f"]:
        train_Y, train_comps = init_train_Y.clone(), init_train_comps.clone()
        within_result = find_max_posterior_mean(outcome_model, train_Y, train_comps)
        within_result.update({"run_id": i, "pe_strategy": pe_strategy})
        within_session_results.append(within_result)

        for j in range(n_check_post_mean):
            train_Y, train_comps = run_pref_learn(
                outcome_model,
                train_Y,
                train_comps,
                n_comps=every_n_comps,
                pe_strategy=pe_strategy,
                verbose=verbose,
            )
            if verbose:
                print(
                    f"Checking posterior mean after {(j+1) * every_n_comps} comps using PE strategy {pe_strategy}"
                )
            within_result = find_max_posterior_mean(
                outcome_model, train_Y, train_comps, verbose=verbose
            )
            within_result.update({"run_id": i, "pe_strategy": pe_strategy})
            within_session_results.append(within_result)

        # Going back to the experimentation stage: generate an additional batch of experimental evaluations
        # with the learned preference model and qNEIUU
        pref_model = fit_pref_model(train_Y, train_comps)
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([NUM_PREF_SAMPLES]))
        pref_obj = LearnedObjective(pref_model=pref_model, sampler=sampler)
        exp_cand_X = gen_exp_cand(outcome_model, pref_obj, q=1, acqf_name="qNEI")
        qneiuu_util = util_func(problem(exp_cand_X)).item()
        print(f"{pe_strategy} qNEIUU candidate utility: {qneiuu_util:.3f}")
        exp_result = {
            "util": qneiuu_util,
            "strategy": pe_strategy,
            "run_id": i,
        }
        exp_candidate_results.append(exp_result)

    # Generate a batch of experimental evaluations using oracle and random baselines
    # True utility
    true_obj = GenericMCObjective(util_func)
    true_obj_cand_X = gen_exp_cand(outcome_model, true_obj, q=1, acqf_name="qNEI")
    true_obj_util = util_func(problem(true_obj_cand_X)).item()
    print(f"True objective utility: {true_obj_util:.3f}")
    exp_result = {
        "util": true_obj_util,
        "strategy": "True Utility",
        "run_id": i,
    }
    exp_candidate_results.append(exp_result)

    # Random experiment
    _, random_Y = generate_random_exp_data(problem, 1)
    random_util = util_func(random_Y).item()
    print(f"Random experiment utility: {random_util:.3f}")
    exp_result = {
        "util": random_util,
        "strategy": "Random Experiment",
        "run_id": i,
    }
    exp_candidate_results.append(exp_result)
    
