#include "PricingEngine.hpp"

#include <vector>
#include <algorithm>
#include <utility>
#include <boost/math/distributions/normal.hpp>

using namespace std;

void OptionPricer::weighted_Laguerre(const Eigen::Ref<const Eigen::ArrayXd> &s, Eigen::Ref<Eigen::MatrixXd> X, int l)
{ // Require l > 0
    X.col(0) = (-0.5 * s).exp();
    if (l > 1)
    {
        X.col(1).array() = X.col(0).array() * (1.0 - s);
    
        for (int i = 2; i < l; i++)
        {
            X.col(i).array() = ((2.0 * i - 1.0 - s) * X.col(i - 1).array() - (i - 1.0) * X.col(i - 2).array()) / i;
        }
    }
}

double OptionPricer::BSM_European(double spot, double maturity)
{
    spot = (spot > 0.0) ? spot : S0;
    maturity = (maturity > 0.0) ? maturity : T;
    double d1 = (log(spot / K) + (r - q + 0.5 * sigma * sigma) * maturity) / (sigma * sqrt(maturity));
    double d2 = d1 - sigma * sqrt(maturity);
    boost::math::normal norm(0.0, 1.0);
    double price = phi() * (spot * exp(-q * maturity) * boost::math::cdf(norm, phi() * d1) - K * exp(-r * maturity) * boost::math::cdf(norm, phi() * d2));
    return price;
}

double OptionPricer::BSM_Continuous_Geometric_Asian()
{
    double q_hat = 0.5 * (r + q + sigma * sigma / 6.0);
    double sigma_hat = sigma / sqrt(3);
    double d1 = (log(S0 / K) + (r - q_hat + 0.5 * sigma_hat * sigma_hat) * T) / (sigma_hat * sqrt(T));
    double d2 = d1 - sigma_hat * sqrt(T);
    boost::math::normal norm(0.0, 1.0);
    double price = phi() * (S0 * exp(-q_hat * T) * boost::math::cdf(norm, phi() * d1) - K * exp(-r * T) * boost::math::cdf(norm, phi() * d2));
    return price;
}

double OptionPricer::BSM_Discrete_Geometric_Asian(int m)
{
    double sigma_G = sigma * sqrt((m + 1.0) * (2.0 * m + 1.0) / (6.0 * m * m));
    double b_G = (m + 1.0) / (2.0 * m) * (r - q - 0.5 * sigma * sigma) + 0.5 * sigma_G * sigma_G;
    double d1 = (log(S0 / K) + (b_G + 0.5 * sigma_G * sigma_G) * T) / (sigma_G * sqrt(T));
    double d2 = d1 - sigma_G * sqrt(T);
    boost::math::normal norm(0.0, 1.0);
    double price = phi() * (S0 * exp((b_G - r) * T) * boost::math::cdf(norm, phi() * d1) - K * exp(-r * T) * boost::math::cdf(norm, phi() * d2));
    return price;
}

double OptionPricer::BSM_Moment_Matching_Arithmatic_Asian(int m)
{
    double dt = T / m; // Time step
    // Precalculation
    double mu = r - q;
    double var = sigma * sigma;
    /* May cause Catastrophic Cancellation
    double u = exp(mu * dt);               // e^{(r-q)dt}
    double v = exp((2.0 * mu + var) * dt); // e^{(2(r-q)+sigma^2)dt}
    double w = exp((mu + var) * dt);       // e^{(r-q+sigma^2)dt}

    // First moment of average asset price
    double M1 = S0 / m * u * (1.0 - pow(u, m)) / (1.0 - u);

    // Second moment of average asset price
    double sum_v = v * (1.0 - pow(v, m)) / (1.0 - v);
    double cross_term = 2.0 / (1.0 - w) * ((u * (u - pow(u, m))) / (1.0 - u) - (w * (w - pow(w, m))) / (1.0 - w));
    double M2 = S0 * S0 / (m * m) * (cross_term + sum_v);
    */

    // First moment of average asset price
    double sum_S = 0.0;
    for (int i = 1; i <= m; i++)
    {
        sum_S += exp(mu * i * dt);
    }
    double M1 = S0 * sum_S / m;

    // Second moment of average asset price
    double sum_cross = 0.0;
    for (int i = 1; i <= m; i++)
    {
        // i == j: E[S(ti)^2]
        sum_cross += exp((2.0 * mu + sigma * sigma) * i * dt);

        // j > i: 2 * E[S(ti) * S(tj)]
        for (int j = i + 1; j <= m; j++)
        {
            sum_cross += 2.0 * exp(mu * (i + j) * dt + sigma * sigma * i * dt);
        }
    }
    double M2 = S0 * S0 * sum_cross / (m * m);

    // Log-normal Moment Matching: ln(A) ~ N(mu_ln, v_ln^2)
    double v_ln = sqrt(log(M2 / (M1 * M1)));
    double mu_ln = log(M1 * M1 / sqrt(M2));

    // Price in BSM style
    double d1 = (mu_ln - log(K) + v_ln * v_ln) / v_ln;
    double d2 = d1 - v_ln;
    boost::math::normal norm(0.0, 1.0);
    double price = phi() * exp(-r * T) * (M1 * boost::math::cdf(norm, phi() * d1) - K * boost::math::cdf(norm, phi() * d2));
    return price;
}

double OptionPricer::CRR_European(int m)
{
    double dt = T / m; // Time step
    double discount = exp(-r * dt);
    // CRR tree: up and down factor, up-tick probability
    double u = exp(sigma * sqrt(dt)), d = 1.0 / u, p = (exp((r - q) * dt) - d) / (u - d);

    std::vector<double> V(m + 1);
    for (int i = 0; i <= m; ++i)
    {
        double Si = S0 * pow(u, 2 * i - m);
        V[i] = max(phi() * (Si - K), 0.0);
    }

    for (int t = m - 1; t >= 0; t--)
    {
        for (int i = 0; i <= t; i++)
        {
            V[i] = discount * (p * V[i + 1] + (1 - p) * V[i]);
        }
    }

    return V[0];
}

double OptionPricer::CRR_American(int m, bool bs)
{
    double dt = T / m; // Time step
    double discount = exp(-r * dt);
    // CRR tree: up and down factor, up-tick probability
    double u = exp(sigma * sqrt(dt)), d = 1.0 / u, p = (exp((r - q) * dt) - d) / (u - d);

    vector<double> V(m + 1);
    if (bs)
    { // Use exact BSM value as continuation value at the penultimate timestep to remove nonlinearity error in the final timestep
        for (int i = 0; i <= m - 1; i++)
        {
            double Si = S0 * pow(u, 2 * i - (m - 1));
            V[i] = max(phi() * (Si - K), BSM_European(Si, dt));
        }
    }
    else
    {
        for (int i = 0; i <= m; i++)
        {
            double Si = S0 * pow(u, 2 * i - m);
            V[i] = max(phi() * (Si - K), 0.0);
        }
    }

    for (int t = bs ? m - 2 : m - 1; t >= 0; t--)
    {
        for (int i = 0; i <= t; i++)
        {
            double Si = S0 * pow(u, 2 * i - t);
            V[i] = max(phi() * (Si - K), discount * (p * V[i + 1] + (1 - p) * V[i]));
        }
    }

    return V[0];
}

double OptionPricer::BBSR_American(int m)
{
    double price_2m = CRR_American(2 * m, true); // BBS price with 2m time steps
    double price_m = CRR_American(m, true);      // BBS price with m time steps
    double price = 2.0 * price_2m - price_m;     // Richardson Extrapolation

    return price;
}

Eigen::ArrayXXd OptionPricer::Brownian_Generator(int n, int m, bool MomentMatch)
{
    // Generate standard Brownian motion
    normal_distribution<> rnorm(0.0, 1.0);
    Eigen::ArrayXXd z = Eigen::ArrayXXd::NullaryExpr(n, m, [&]()
                                                     { return rnorm(gen); });
    if (MomentMatch)
    {
        for (int j = 0; j < m; ++j)
        {
            double mean = z.col(j).mean();
            double var  = (z.col(j) - mean).square().sum() / (n - 1);
            z.col(j) = (z.col(j) - mean) / sqrt(var);
        }
    }
    return z;
}

pair<Eigen::ArrayXXd, optional<Eigen::ArrayXXd>>
OptionPricer::Path_Simulation(int n, int m, bool Antithetic, bool EMS, bool MomentMatch)
{
    Eigen::ArrayXXd z = Brownian_Generator(n, m, MomentMatch);
    double dt = T / m; // Time step
    double drift = (r - q - 0.5 * sigma * sigma) * dt;
    double diffusion = sigma * sqrt(dt);

    // Simulate asset price at each time step
    Eigen::ArrayXXd S1(n, m + 1);
    S1.col(0).setConstant(S0);
    Eigen::ArrayXXd log_ret1 = drift + diffusion * z;
    double step_growth = exp((r - q) * dt);
    double expected_S = S0;
    for (int j = 0; j < m; j++)
    {
        S1.col(j + 1) = S1.col(j) * log_ret1.col(j).exp();
        if (EMS)
        { // Duan & Simonato (1998) Empirical Martingale Simulation
            expected_S *= step_growth;
            S1.col(j + 1) *= expected_S / S1.col(j + 1).mean();
        }
    }

    // Simulate antithetic price paths
    optional<Eigen::ArrayXXd> S2 = nullopt;
    if (Antithetic)
    {
        S2.emplace(n, m + 1);
        S2->col(0).setConstant(S0);
        Eigen::ArrayXXd log_ret2 = drift - diffusion * z;
        expected_S = S0;
        for (int j = 0; j < m; j++)
        {
            S2->col(j + 1) = S2->col(j) * log_ret2.col(j).exp();
            if (EMS)
            {
                expected_S *= step_growth;
                S2->col(j + 1) *= expected_S / S2->col(j + 1).mean();
            }
        }
    }
    return {std::move(S1), std::move(S2)};
}

MonteCarloResult OptionPricer::Monte_Carlo_European(int n, bool Antithetic, bool EMS, bool MomentMatch)
{
    auto start = chrono::high_resolution_clock::now();

    // Simulate stock price at each time step
    auto [S, S_a] = Path_Simulation(n, 1, Antithetic, EMS, MomentMatch);

    // Simulate option payoffs
    Eigen::ArrayXd payoff = (phi() * (S.col(1) - K)).max(0.0);
    if (Antithetic)
    {
        Eigen::ArrayXd payoff_a = (phi() * (S_a->col(1) - K)).max(0.0);
        payoff = (payoff + payoff_a) * 0.5;
    }
    double discount = exp(-r * T);

    chrono::duration<double> duration = chrono::high_resolution_clock::now() - start;

    return MonteCarloResult(payoff, duration, discount);
}

MonteCarloResult OptionPricer::Monte_Carlo_Asian(int n, int m, bool Antithetic, bool EMS, bool MomentMatch, bool CtrlVar)
{
    auto start = chrono::high_resolution_clock::now();

    // Simulate stock price at each time step
    auto [S, S_a] = Path_Simulation(n, m, Antithetic, EMS, MomentMatch);

    // Simulate option payoffs
    Eigen::ArrayXd payoff = (phi() * (S.rightCols(m).rowwise().mean() - K)).max(0.0);
    if (Antithetic)
    {
        Eigen::ArrayXd payoff_a = (phi() * (S_a->rightCols(m).rowwise().mean() - K)).max(0.0);
        payoff = (payoff + payoff_a) * 0.5;
    }
    double discount = exp(-r * T);

    if (CtrlVar)
    {
        // Simulate geometric Asian option payoffs
        Eigen::ArrayXd payoff_G = (phi() * (S.rightCols(m).log().rowwise().mean().exp() - K)).max(0.0); // Avoid overflow
        if (Antithetic)
        {
            Eigen::ArrayXd payoff_G_a = (phi() * (S_a->rightCols(m).log().rowwise().mean().exp() - K)).max(0.0);
            payoff_G = (payoff_G + payoff_G_a) * 0.5;
        }
        // Calculate analytic geometric Asian call option prices
        double BSM_GeoAs_payoff = BSM_Discrete_Geometric_Asian(m) / discount;

        // (Regression) Optimal control coefficient
        Eigen::ArrayXd Y_centered = payoff - payoff.mean();
        Eigen::ArrayXd X_centered = payoff_G - payoff_G.mean();
        double var_X = X_centered.square().mean();
        if (var_X > 1e-12)
        {
            double lambda = (Y_centered * X_centered).mean() / var_X;
            payoff = payoff + lambda * (BSM_GeoAs_payoff - payoff_G);
        }
    }

    chrono::duration<double> duration = chrono::high_resolution_clock::now() - start;

    return MonteCarloResult(payoff, duration, discount);
}

MonteCarloResult OptionPricer::Least_Squares_Monte_Carlo_American(int n, int m, int l)
{
    auto start = chrono::high_resolution_clock::now();

    // Simulate stock price at each time step
    Eigen::ArrayXXd S = Path_Simulation(n, m).first;

    // Memory pre-allocation
    vector<int> in_the_money(n);
    Eigen::ArrayXd excercise_value(n);
    Eigen::ArrayXd scaled_S(n); // Renormalize to avoid underflows in basis functions
    Eigen::VectorXd Y(n);
    Eigen::MatrixXd X(n, l);
    Eigen::VectorXd continuation_value(n);
    // Calculate option price at each time step
    double dt = T / m;
    double discount = exp(-r * dt);
    Eigen::ArrayXd current_value = (phi() * (S.col(m) - K)).max(0.0);
    for (int t = m - 1; t >= 1; t--)
    {
        current_value *= discount;
        // Calculate option value if exercise at t
        excercise_value = (phi() * (S.col(t) - K)).max(0.0);

        // Extract in-the-money path
        int n_itm = 0;
        for (int i = 0; i < n; i++)
        {
            if (excercise_value[i] > 0.0)
            {
                in_the_money[n_itm] = i;
                scaled_S[n_itm] = S(i, t) / K;
                Y[n_itm] = current_value[i];
                n_itm++;
            }
        }

        if (n_itm > 0)
        { // For in-the-money path, run Least Squares Monte Carlo
            if (n_itm < l)
            { // If number of itm paths is smaller then the number of calibration functions then early exercise if exercise_value > 0
                continuation_value.head(n_itm).setZero();
            }
            else
            {
                // Eigen::Map<Eigen::ArrayXd> scaled_S_itm(scaled_S.data(), n_itm);
                // Eigen::Map<Eigen::ArrayXXd, 0, Eigen::OuterStride<>> X_itm(X.data(), n_itm, l, Eigen::OuterStride<>(X.rows()));
                // Eigen::Map<Eigen::VectorXd> Y_itm(Y.data(), n_itm);
                // Calculate regressors
                weighted_Laguerre(scaled_S.head(n_itm), X.topRows(n_itm), l);
                // Regress on in-the-money path
                Eigen::VectorXd para = X.topRows(n_itm).householderQr().solve(Y.head(n_itm)); // QR decomposition
                // Calculate continuation value
                continuation_value.head(n_itm) = X.topRows(n_itm) * para;
            }
            // Compare excercise value with continuation value
            for (int i = 0; i < n_itm; i++)
            {
                if (excercise_value[in_the_money[i]] > continuation_value[i])
                {
                    current_value[in_the_money[i]] = excercise_value[in_the_money[i]];
                }
            }
        }
    }
    current_value = (discount * current_value).max(phi() * (S0 - K));

    chrono::duration<double> duration = chrono::high_resolution_clock::now() - start;

    return MonteCarloResult(current_value, duration, 1);
}
