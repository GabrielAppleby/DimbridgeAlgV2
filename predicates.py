def create_predicates(mu, a, mean, scale, feature_min, feature_max):
    r = 1 / a.abs()
    predicates = []
    for feature_idx in range(mean.shape[0]):
        r_k = (r[feature_idx] * scale[feature_idx]).item()
        mu_k = (mu[feature_idx] * scale[feature_idx] + mean[feature_idx]).item()
        ci = [mu_k - r_k, mu_k + r_k]
        assert ci[0] < ci[1], "ci[0] is not less than ci[1]"
        if ci[0] < feature_min[feature_idx]:
            ci[0] = feature_min[feature_idx]
        if ci[1] > feature_max[feature_idx]:
            ci[1] = feature_max[feature_idx]
        should_include = not (
            ci[0] <= feature_min[feature_idx] and ci[1] >= feature_max[feature_idx]
        )
        if should_include:
            predicates.append(dict(dim=feature_idx, interval=ci))
    for p in predicates:
        print(p)
    print("next")
