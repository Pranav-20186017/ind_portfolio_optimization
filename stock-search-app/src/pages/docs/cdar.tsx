import React from 'react';
import {
  Typography,
  Container,
  Box,
  Paper,
  Grid,
  Button,
  Link as MuiLink
} from '@mui/material';
import Link from 'next/link';
import Head from 'next/head';
import 'katex/dist/katex.min.css';
import { InlineMath, BlockMath } from 'react-katex';

// Re‑usable equation wrapper for consistent styling
const Equation = ({ math }: { math: string }) => (
  <Box sx={{ p: 2, bgcolor: '#f5f5f5', borderRadius: 1, my: 2, textAlign: 'center' }}>
    <BlockMath math={math} />
  </Box>
);

const ConditionalDrawdownAtRiskPage: React.FC = () => {
  return (
    <>
      <Head>
        <title>Conditional Drawdown at Risk (CDaR) | Portfolio Optimization</title>
        <meta
          name="description"
          content="Learn about Conditional Drawdown at Risk (CDaR), the expected value of drawdown when exceeding the Drawdown at Risk threshold, measuring tail drawdown risk."
        />
      </Head>

      <Container maxWidth="lg" sx={{ py: 4 }}>
        {/* Navigation */}
        <Box sx={{ mb: 4, display: 'flex', gap: 2 }}>
          <Link href="/docs" passHref>
            <Button variant="outlined" color="primary">← Back to Education</Button>
          </Link>
          <Link href="/" passHref>
            <Button variant="outlined" color="secondary">← Back to Portfolio Optimizer</Button>
          </Link>
        </Box>

        {/* Title */}
        <Box sx={{ mb: 6, textAlign: 'center' }}>
          <Typography variant="h3" component="h1" gutterBottom>
            Conditional Drawdown at Risk (CDaR)
          </Typography>
          <Typography variant="subtitle1" color="text.secondary">
            The expected value of drawdown when exceeding the Drawdown at Risk threshold, measuring tail drawdown risk
          </Typography>
        </Box>

        {/* Overview */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>Overview</Typography>
          <Typography paragraph>
            <strong>Conditional Drawdown at Risk (CDaR)</strong> is an advanced risk metric that extends the concept of Drawdown at Risk (DaR) by focusing on the severity of extreme drawdowns. While DaR tells us the maximum drawdown level that won't be exceeded with a certain confidence, CDaR measures the expected magnitude of drawdowns that exceed this threshold.
          </Typography>
          <Typography paragraph>
            CDaR provides deeper insight into tail risk by quantifying the average loss in the worst-case scenarios, making it particularly valuable for risk-averse investors and those concerned with protecting against catastrophic market events. It belongs to the family of coherent risk measures, satisfying mathematical properties that ensure consistent risk assessment across different portfolios and market conditions.
          </Typography>
          <Typography paragraph>
            This metric is especially relevant for investors with limited risk tolerance, endowments with spending constraints, and pension funds with specific funding requirements, as it helps quantify the expected severity of extreme drawdowns that could threaten financial stability or require significant strategy adjustments.
          </Typography>
        </Paper>

        {/* Intuitive Explanation */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>Intuitive Explanation</Typography>
          <Typography paragraph>
            Imagine you're planning an expedition in a mountainous region. Drawdown at Risk (DaR) would tell you, "With 95% confidence, the maximum elevation drop you should encounter won't exceed 1,000 feet." This is valuable information, but it doesn't tell you what to expect if you do face a drop exceeding 1,000 feet.
          </Typography>
          <Typography paragraph>
            Conditional Drawdown at Risk (CDaR) addresses this gap by saying, "If you do encounter drops exceeding 1,000 feet, the average of those extreme drops will be 1,300 feet." This additional information is crucial for proper preparation and risk management in case you face these extreme scenarios.
          </Typography>
          <Box sx={{ bgcolor: 'background.paper', p: 2, borderRadius: 1, border: '1px solid #e0e0e0', mt: 2 }}>
            <Typography variant="body1">
              <strong>Real-world analogy:</strong> A coastal city's flood management team uses DaR to know that 95% of the time, flooding won't exceed 3 feet. However, for comprehensive disaster planning, they need to know CDaR — the average severity of those rare floods that do exceed 3 feet. If the CDaR is 5 feet, the city needs to design infrastructure and emergency responses to handle these more extreme but still possible scenarios.
            </Typography>
          </Box>
        </Paper>

        {/* Mathematical Explanation */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>Detailed Mathematical Explanation</Typography>

          <Typography paragraph>
            To define Conditional Drawdown at Risk, we first need to recall the concepts of drawdown and Drawdown at Risk (DaR).
          </Typography>

          <Typography paragraph>
            For a time series of portfolio values <InlineMath math="P_t" />, the drawdown at time t is defined as:
          </Typography>

          <Box sx={{ p: 3, bgcolor: 'background.paper', borderRadius: 1, border: '1px solid #e0e0e0', my: 3 }}>
            <Typography variant="subtitle1" gutterBottom><strong>Drawdown Formula</strong></Typography>
            <Equation math="DD_t = \frac{M_t - P_t}{M_t}" />
            <Typography variant="body2">
              where <InlineMath math="M_t = \max_{s \leq t} P_s" /> is the maximum portfolio value up to time t.
            </Typography>
          </Box>

          <Typography paragraph>
            The Drawdown at Risk (DaR) at confidence level α is defined as:
          </Typography>

          <Equation math="DaR_\alpha = \inf \{ x \in \mathbb{R} : P(MDD \leq x) \geq \alpha \}" />

          <Typography paragraph>
            Building on these foundations, Conditional Drawdown at Risk (CDaR) at confidence level α is formally defined as:
          </Typography>

          <Box sx={{ p: 3, bgcolor: 'background.paper', borderRadius: 1, border: '1px solid #e0e0e0', my: 3 }}>
            <Typography variant="subtitle1" gutterBottom><strong>Conditional Drawdown at Risk Formula</strong></Typography>
            <Equation math="CDaR_\alpha(X) = \frac{1}{1-\alpha} \int_{\alpha}^{1} DaR_u(X) du" />
            <Typography variant="body2">
              where X represents the portfolio, α is the confidence level, and <InlineMath math="DaR_u(X)" /> is the Drawdown at Risk at confidence level u.
            </Typography>
          </Box>

          <Typography paragraph>
            Alternatively, CDaR can be expressed as the conditional expectation of the maximum drawdown (MDD) given that it exceeds the DaR threshold:
          </Typography>

          <Equation math="CDaR_\alpha = \mathbb{E}[MDD \mid MDD > DaR_\alpha]" />

          <Typography paragraph>
            For empirical calculation with a discrete set of drawdown observations, CDaR can be computed as:
          </Typography>

          <Equation math="CDaR_\alpha = DaR_\alpha + \frac{1}{(1-\alpha)n} \sum_{i: MDD_i > DaR_\alpha} (MDD_i - DaR_\alpha)" />

          <Typography variant="h6" gutterBottom sx={{ mt: 3 }}>Properties of CDaR</Typography>
          <Typography paragraph>
            CDaR possesses several important mathematical properties:
          </Typography>
          <ul>
            <li>
              <Typography paragraph>
                <strong>Coherence:</strong> Unlike DaR, CDaR is a coherent risk measure, satisfying the properties of monotonicity, sub-additivity, homogeneity, and translation invariance.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Sub-additivity:</strong> For any two portfolios X and Y, CDaR₍ₓ₊ᵧ₎ ≤ CDaRₓ + CDaRᵧ, which means diversification doesn't increase risk.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Convexity:</strong> CDaR is a convex function, making it suitable for convex optimization techniques commonly used in portfolio construction.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Consistency with stochastic dominance:</strong> If one portfolio stochastically dominates another in terms of drawdown distribution, CDaR will appropriately rank their risks.
              </Typography>
            </li>
          </ul>
        </Paper>

        {/* Implementation Details */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>Implementation in Portfolio Optimization</Typography>
          <Typography paragraph>
            CDaR can be integrated into portfolio optimization frameworks in several powerful ways:
          </Typography>
          <ul>
            <li>
              <Typography paragraph>
                <strong>Risk Minimization:</strong> Portfolios can be constructed to minimize CDaR for a given expected return, leading to more robustness against extreme drawdowns.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Constraint-Based Approach:</strong> CDaR can be used as a constraint in optimization problems, ensuring that the portfolio's tail drawdown risk stays below a specified threshold.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Multi-Objective Optimization:</strong> CDaR can be combined with other objectives like expected return or Sharpe ratio in Pareto-efficient portfolio construction.
              </Typography>
            </li>
          </ul>
          <Typography paragraph>
            Our implementation calculates CDaR through the following process:
          </Typography>
          <ol>
            <li>
              <Typography paragraph>
                <strong>Historical Data Analysis:</strong> We collect historical returns for all assets in the portfolio.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Portfolio Path Simulation:</strong> For each scenario (historical or Monte Carlo), we calculate the cumulative portfolio value path and identify all drawdowns.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>DaR Calculation:</strong> We determine the α-quantile of the maximum drawdown distribution to establish the DaR threshold.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>CDaR Calculation:</strong> We compute the average of all maximum drawdowns that exceed the DaR threshold.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Optimization:</strong> When used for portfolio optimization, we employ convex optimization techniques that efficiently handle CDaR as either an objective function or a constraint.
              </Typography>
            </li>
          </ol>
          <Box sx={{ p: 2, bgcolor: 'background.paper', borderRadius: 1, border: '1px solid #e0e0e0', mt: 2, textAlign: 'center' }}>
            <Typography variant="subtitle2" gutterBottom>CDaR Visualization (Placeholder)</Typography>
            <Box sx={{ height: '300px', bgcolor: '#f5f5f5', display: 'flex', alignItems: 'center', justifyContent: 'center', mb: 2 }}>
              <Typography variant="body2" color="text.secondary">
                [Placeholder for drawdown distribution chart showing DaR threshold and CDaR calculation]
              </Typography>
            </Box>
            <Typography variant="caption" color="text.secondary">
              This chart illustrates the distribution of maximum drawdowns, with the DaR threshold at the 95% confidence level and CDaR representing the average of drawdowns in the right tail.
            </Typography>
          </Box>
        </Paper>

        {/* Worked Example */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>Worked Example</Typography>
          <Typography paragraph>
            Let's calculate the Conditional Drawdown at Risk for a portfolio using a simplified example:
          </Typography>

          <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>Step 1: Historical Maximum Drawdowns</Typography>
          <Typography paragraph>
            Suppose we have 20 observations of annual maximum drawdowns for a portfolio (in percentage):
          </Typography>
          <Typography paragraph>
            8%, 12%, 5%, 15%, 7%, 22%, 10%, 13%, 9%, 18%, 11%, 14%, 6%, 19%, 25%, 16%, 20%, 17%, 34%, 28%
          </Typography>

          <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>Step 2: Calculate DaR at 90% confidence level</Typography>
          <Typography paragraph>
            We first sort the drawdowns in ascending order:
          </Typography>
          <Typography paragraph>
            5%, 6%, 7%, 8%, 9%, 10%, 11%, 12%, 13%, 14%, 15%, 16%, 17%, 18%, 19%, 20%, 22%, 25%, 28%, 34%
          </Typography>
          <Typography paragraph>
            For a 90% confidence level, we identify the 90th percentile, which is the 18th value in our sorted list of 20 observations:
          </Typography>
          <Typography paragraph>
            DaR₉₀% = 25%
          </Typography>
          <Typography paragraph>
            This means that with 90% confidence, the maximum drawdown should not exceed 25%.
          </Typography>

          <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>Step 3: Calculate CDaR at 90% confidence level</Typography>
          <Typography paragraph>
            Now, we calculate the average of the drawdowns that exceed the DaR₉₀% threshold of 25%:
          </Typography>
          <Typography paragraph>
            Drawdowns exceeding 25%: 28%, 34%
          </Typography>
          <Typography paragraph>
            CDaR₉₀% = (28% + 34%) / 2 = 31%
          </Typography>

          <Typography paragraph>
            Alternatively, using the formula that incorporates the DaR value:
          </Typography>
          <Typography paragraph>
            CDaR₉₀% = 25% + (1/(1-0.9) × 20) × [(28% - 25%) + (34% - 25%)] = 25% + (1/2) × [3% + 9%] = 25% + 6% = 31%
          </Typography>

          <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>Step 4: Interpretation</Typography>
          <Typography paragraph>
            The results tell us:
          </Typography>
          <ul>
            <li>
              <Typography paragraph>
                With 90% confidence, the maximum drawdown won't exceed 25% (this is the DaR).
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                In the 10% worst cases where the drawdown does exceed 25%, the average drawdown will be 31% (this is the CDaR).
              </Typography>
            </li>
          </ul>
          <Typography paragraph>
            This information is particularly valuable for stress testing and preparing contingency plans for extreme market conditions. An investor might decide that while they can tolerate a 25% drawdown, the prospect of an average 31% drawdown in worst-case scenarios exceeds their risk tolerance, prompting them to adjust their portfolio allocation.
          </Typography>
        </Paper>

        {/* Practical Applications */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>Practical Applications</Typography>
          
          <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>Risk-Constrained Portfolio Optimization</Typography>
          <Typography paragraph>
            CDaR serves as an effective risk constraint in portfolio optimization. By setting a maximum acceptable CDaR level, portfolio managers can ensure that the expected drawdown severity in worst-case scenarios remains within tolerable limits while maximizing expected returns.
          </Typography>
          
          <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>Stress Testing and Scenario Analysis</Typography>
          <Typography paragraph>
            Financial institutions use CDaR to stress test their portfolios against extreme market conditions. By estimating CDaR under various historical and hypothetical scenarios, risk managers gain insight into the potential magnitude of losses during crisis periods.
          </Typography>
          
          <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>Hedge Fund Evaluation</Typography>
          <Typography paragraph>
            CDaR provides a valuable metric for evaluating and comparing hedge funds, particularly those claiming to offer downside protection. A lower CDaR indicates better management of tail risk and potentially more robust performance during market downturns.
          </Typography>
          
          <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>Capital Allocation and Risk Budgeting</Typography>
          <Typography paragraph>
            CDaR enables more sophisticated risk budgeting by focusing on the allocation of risk to different portfolio components based on their contribution to the portfolio's tail drawdown risk, rather than their volatility contribution.
          </Typography>
          
          <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>Derivatives Pricing and Hedging</Typography>
          <Typography paragraph>
            CDaR can inform the pricing and design of derivatives aimed at providing protection against sustained market downturns, such as options on maximum drawdown or insurance products that pay out based on drawdown levels.
          </Typography>
        </Paper>

        {/* Advantages and Limitations */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>Advantages and Limitations</Typography>
          
          <Typography variant="h6" gutterBottom>Advantages</Typography>
          <ul>
            <li>
              <Typography paragraph>
                <strong>Coherent Risk Measure:</strong> Unlike DaR, CDaR is a coherent risk measure, satisfying mathematical properties that ensure consistent risk assessment, particularly the sub-additivity property that captures diversification benefits.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Tail Risk Focus:</strong> CDaR provides detailed information about the severity of extreme drawdowns, offering deeper insight into tail risk than metrics that only capture the threshold (like DaR) or the average risk (like standard deviation).
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Path Dependency:</strong> By incorporating the temporal sequence of returns, CDaR captures dynamic risk aspects that point-in-time measures miss, making it more representative of the actual investor experience.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Optimization-Friendly:</strong> CDaR's convexity makes it suitable for efficient optimization algorithms, enabling practical implementation in portfolio construction processes.
              </Typography>
            </li>
          </ul>
          
          <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>Limitations</Typography>
          <ul>
            <li>
              <Typography paragraph>
                <strong>Data Requirements:</strong> Accurate CDaR estimation requires extensive historical data or sophisticated simulation techniques to adequately capture the distribution of extreme drawdowns.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Computational Complexity:</strong> Computing CDaR for complex portfolios, especially when used in optimization procedures, can be computationally intensive and may require specialized algorithms.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Estimation Uncertainty:</strong> The estimation of CDaR is subject to significant uncertainty due to the limited number of extreme events in historical data, potentially leading to estimation errors.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Interpretation Challenges:</strong> CDaR may be less intuitive to non-technical stakeholders compared to simpler metrics like maximum drawdown, potentially requiring additional explanation and education.
              </Typography>
            </li>
          </ul>
        </Paper>

        {/* Related Metrics */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>Related Metrics</Typography>
          <Typography paragraph>
            CDaR belongs to a family of risk measures that focus on different aspects of drawdown risk:
          </Typography>
          <ul>
            <li>
              <Typography paragraph>
                <strong>Drawdown at Risk (DaR):</strong> The threshold value that maximum drawdown will not exceed with a specified confidence level, serving as the foundation for CDaR calculation.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Maximum Drawdown (MDD):</strong> The largest peak-to-trough decline experienced by a portfolio over a specific time period, representing the worst historical loss from a previous peak.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Average Drawdown (ADD):</strong> The mean of all drawdowns over a time period, providing a measure of the typical drawdown magnitude regardless of severity.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Conditional Value at Risk (CVaR):</strong> A related concept that measures the expected loss exceeding the Value at Risk threshold, but applied to returns at specific points rather than drawdown paths.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Ulcer Index:</strong> A measure that captures both the depth and duration of drawdowns by calculating the square root of the mean of the squared drawdowns.
              </Typography>
            </li>
          </ul>
        </Paper>

        {/* References */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>References</Typography>
          <ul>
            <li>
              <Typography paragraph>
                Chekhlov, A., Uryasev, S., & Zabarankin, M. (2005). "Drawdown Measure in Portfolio Optimization." International Journal of Theoretical and Applied Finance, 8(01), 13-58.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                Goldberg, L. R., & Mahmoud, O. (2017). "Drawdown: From Practice to Theory and Back Again." Mathematics and Financial Economics, 11(3), 275-297.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                Alexander, G. J., & Baptista, A. M. (2006). "Portfolio Selection with Drawdown Constraints." Journal of Banking & Finance, 30(11), 3171-3189.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                Zabarankin, M., Pavlikov, K., & Uryasev, S. (2014). "Capital Asset Pricing Model (CAPM) with Drawdown Measure." European Journal of Operational Research, 234(2), 508-517.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                Rockafellar, R. T., & Uryasev, S. (2002). "Conditional Value-at-Risk for General Loss Distributions." Journal of Banking & Finance, 26(7), 1443-1471.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                Lohre, H., Nessler, T., & Ottink, J. (2014). "Drawdown Diversification: A Risk Management Approach to Improved Risk-Adjusted Returns." Journal of Portfolio Management, 40(3), 78-90.
              </Typography>
            </li>
          </ul>
        </Paper>
      </Container>
    </>
  );
};

// Add getStaticProps to generate this page at build time
export const getStaticProps = async () => {
  return {
    props: {},
  };
};

export default ConditionalDrawdownAtRiskPage;
