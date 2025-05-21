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

const DrawdownAtRiskPage: React.FC = () => {
  return (
    <>
      <Head>
        <title>Drawdown at Risk (DaR) | Portfolio Optimization</title>
        <meta
          name="description"
          content="Learn about Drawdown at Risk (DaR), a risk metric representing the maximum expected drawdown that won't be exceeded with a certain confidence level."
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
            Drawdown at Risk (DaR)
          </Typography>
          <Typography variant="subtitle1" color="text.secondary">
            A risk metric representing the maximum expected drawdown that won't be exceeded with a certain confidence level
          </Typography>
        </Box>

        {/* Overview */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>Overview</Typography>
          <Typography paragraph>
            <strong>Drawdown at Risk (DaR)</strong> is an advanced risk metric designed to quantify portfolio risk from a perspective that's highly relevant to investors: the magnitude of potential drawdowns. While traditional risk measures like volatility focus on the dispersion of returns, DaR specifically addresses the downside risk of cumulative losses from peak to trough.
          </Typography>
          <Typography paragraph>
            DaR represents the maximum drawdown that an investment is not expected to exceed with a given confidence level (typically 95% or 99%). It allows investors to make statements like "With 95% confidence, the maximum drawdown of this portfolio should not exceed 25%."
          </Typography>
          <Typography paragraph>
            This metric is particularly valuable for risk-averse investors, wealth managers, and financial advisors who need to set realistic expectations about potential losses during market downturns and ensure that portfolios are aligned with clients' risk tolerance levels.
          </Typography>
        </Paper>

        {/* Intuitive Explanation */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>Intuitive Explanation</Typography>
          <Typography paragraph>
            Imagine you're planning a mountain hiking trip and are concerned about potential elevation drops. Value at Risk (VaR) might tell you how far you might drop on a single section of the trail. But what you're really concerned about is the total descent from the highest peak to the lowest point during your journey—that's what drawdown measures.
          </Typography>
          <Typography paragraph>
            Drawdown at Risk takes this concept further by telling you, "Based on historical data and with 95% confidence, the maximum elevation drop you should expect during your journey will not exceed X feet." This information allows you to prepare appropriately for the descent without being caught off guard.
          </Typography>
          <Box sx={{ bgcolor: 'background.paper', p: 2, borderRadius: 1, border: '1px solid #e0e0e0', mt: 2 }}>
            <Typography variant="body1">
              <strong>Real-world analogy:</strong> A cruise ship operator needs to know not just the average wave height (volatility) but the maximum trough-to-peak difference they might encounter during a voyage (drawdown). DaR at 95% confidence level tells them, "Based on historical data for this route and season, in 95% of voyages, the maximum wave height differential won't exceed 30 feet." This allows them to prepare appropriately and set passenger expectations.
            </Typography>
          </Box>
        </Paper>

        {/* Mathematical Explanation */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>Detailed Mathematical Explanation</Typography>

          <Typography paragraph>
            To understand Drawdown at Risk, we first need to define drawdown. For a time series of portfolio values or returns <InlineMath math="P_t" />, the drawdown at time t is:
          </Typography>

          <Box sx={{ p: 3, bgcolor: 'background.paper', borderRadius: 1, border: '1px solid #e0e0e0', my: 3 }}>
            <Typography variant="subtitle1" gutterBottom><strong>Drawdown Formula</strong></Typography>
            <Equation math="DD_t = \frac{M_t - P_t}{M_t}" />
            <Typography variant="body2">
              where <InlineMath math="M_t = \max_{s \leq t} P_s" /> is the maximum portfolio value up to time t.
            </Typography>
          </Box>

          <Typography paragraph>
            The maximum drawdown over a time period [0,T] is:
          </Typography>

          <Equation math="MDD = \max_{t \in [0,T]} DD_t" />

          <Typography paragraph>
            Based on this foundation, Drawdown at Risk (DaR) at confidence level α is defined as:
          </Typography>

          <Box sx={{ p: 3, bgcolor: 'background.paper', borderRadius: 1, border: '1px solid #e0e0e0', my: 3 }}>
            <Typography variant="subtitle1" gutterBottom><strong>Drawdown at Risk Formula</strong></Typography>
            <Equation math="DaR_\alpha = \inf \{ x \in \mathbb{R} : P(MDD \leq x) \geq \alpha \}" />
            <Typography variant="body2">
              where α is the confidence level (e.g., 0.95 or 0.99).
            </Typography>
          </Box>

          <Typography paragraph>
            In simpler terms, DaR is the quantile of the maximum drawdown distribution at the specified confidence level α. This means that with probability α, the maximum drawdown will not exceed the DaR value.
          </Typography>

          <Typography variant="h6" gutterBottom sx={{ mt: 3 }}>Empirical Calculation</Typography>
          <Typography paragraph>
            To calculate DaR empirically from historical data:
          </Typography>
          <ol>
            <li>
              <Typography paragraph>
                Generate a time series of portfolio values or cumulative returns.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                Calculate the drawdown series using the formula above.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                Find the maximum drawdown for the entire period.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                Repeat steps 1-3 for multiple time periods or using bootstrapping/simulation techniques to generate a distribution of maximum drawdowns.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                Calculate the α-quantile of this distribution, which gives the DaR.
              </Typography>
            </li>
          </ol>

          <Typography variant="h6" gutterBottom sx={{ mt: 3 }}>Properties of DaR</Typography>
          <Typography paragraph>
            DaR has several important properties that make it a useful risk measure:
          </Typography>
          <ul>
            <li>
              <Typography paragraph>
                <strong>Path dependency:</strong> Unlike VaR and CVaR, DaR captures the temporal dimension of risk by accounting for the sequence of returns, not just their distribution.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Investment horizon sensitivity:</strong> DaR increases with the investment horizon, reflecting the greater potential for large drawdowns over longer periods.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Non-normal distribution awareness:</strong> DaR does not assume normally distributed returns and can capture risks from fat tails and serial correlation.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Intuitiveness:</strong> DaR is expressed in the same units as returns (percentage) and relates directly to investor experience, making it easier to interpret than abstract statistical measures.
              </Typography>
            </li>
          </ul>
        </Paper>

        {/* Implementation Details */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>Implementation in Portfolio Optimization</Typography>
          <Typography paragraph>
            DaR can be integrated into portfolio optimization in several ways:
          </Typography>
          <ul>
            <li>
              <Typography paragraph>
                <strong>Risk Constraint:</strong> Portfolios can be optimized to achieve maximum expected return while ensuring that the DaR does not exceed a specified threshold.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Risk Minimization:</strong> For a given expected return level, portfolios can be constructed to minimize the DaR.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Performance Evaluation:</strong> DaR can be used alongside other metrics to evaluate and compare the risk-adjusted performance of different portfolios.
              </Typography>
            </li>
          </ul>
          <Typography paragraph>
            In our implementation, we calculate DaR through the following process:
          </Typography>
          <ol>
            <li>
              <Typography paragraph>
                <strong>Historical Data Analysis:</strong> We collect historical returns for all assets in the portfolio.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Scenario Generation:</strong> We either use historical scenarios or generate Monte Carlo simulations based on estimated parameters.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Portfolio Path Simulation:</strong> For each scenario, we calculate the cumulative portfolio value path and identify the maximum drawdown.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>DaR Calculation:</strong> We determine the α-quantile of the maximum drawdown distribution across all scenarios.
              </Typography>
            </li>
          </ol>
          <Box sx={{ p: 2, bgcolor: 'background.paper', borderRadius: 1, border: '1px solid #e0e0e0', mt: 2, textAlign: 'center' }}>
            <Typography variant="subtitle2" gutterBottom>Drawdown at Risk Visualization (Placeholder)</Typography>
            <Box sx={{ height: '300px', bgcolor: '#f5f5f5', display: 'flex', alignItems: 'center', justifyContent: 'center', mb: 2 }}>
              <Typography variant="body2" color="text.secondary">
                [Placeholder for drawdown distribution chart with DaR highlighted]
              </Typography>
            </Box>
            <Typography variant="caption" color="text.secondary">
              The chart illustrates a distribution of maximum drawdowns with DaR at 95% and 99% confidence levels marked.
            </Typography>
          </Box>
        </Paper>

        {/* Worked Example */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>Worked Example</Typography>
          <Typography paragraph>
            Let's consider a simplified example to illustrate how DaR is calculated:
          </Typography>

          <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>Step 1: Historical Data</Typography>
          <Typography paragraph>
            Suppose we have 5 years of monthly data for a portfolio, and we compute the maximum drawdown for each year:
          </Typography>
          <Typography component="div" sx={{ mb: 2 }}>
            <ul>
              <li>Year 1: Maximum Drawdown = 15%</li>
              <li>Year 2: Maximum Drawdown = 8%</li>
              <li>Year 3: Maximum Drawdown = 22%</li>
              <li>Year 4: Maximum Drawdown = 12%</li>
              <li>Year 5: Maximum Drawdown = 18%</li>
            </ul>
          </Typography>

          <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>Step 2: Calculate DaR at 80% confidence level</Typography>
          <Typography paragraph>
            We arrange the maximum drawdowns in ascending order:
          </Typography>
          <Typography paragraph>
            8%, 12%, 15%, 18%, 22%
          </Typography>
          <Typography paragraph>
            For an 80% confidence level, we need the 80th percentile of this distribution. With 5 data points, the 80th percentile is approximately the 4th value:
          </Typography>
          <Typography paragraph>
            DaR₈₀% = 18%
          </Typography>
          <Typography paragraph>
            This means that with 80% confidence, the maximum drawdown in any given year should not exceed 18%.
          </Typography>

          <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>Step 3: Calculate DaR at 95% confidence level</Typography>
          <Typography paragraph>
            For a 95% confidence level with only 5 data points, we would typically use the highest observed value as an approximation:
          </Typography>
          <Typography paragraph>
            DaR₉₅% ≈ 22%
          </Typography>
          <Typography paragraph>
            This is a simplified approach. In practice, with more data points, we would determine the exact 95th percentile.
          </Typography>

          <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>Step 4: Interpretation</Typography>
          <Typography paragraph>
            Based on our historical data:
          </Typography>
          <ul>
            <li>
              <Typography paragraph>
                With 80% confidence, we expect the annual maximum drawdown not to exceed 18%.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                With 95% confidence, we expect the annual maximum drawdown not to exceed 22%.
              </Typography>
            </li>
          </ul>
          <Typography paragraph>
            An investor can use this information to assess whether the portfolio's risk profile aligns with their risk tolerance and investment objectives.
          </Typography>
        </Paper>

        {/* Practical Applications */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>Practical Applications</Typography>
          <Typography paragraph>
            Drawdown at Risk is particularly useful in the following contexts:
          </Typography>
          
          <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>Risk Management</Typography>
          <Typography paragraph>
            DaR helps portfolio managers set stop-loss levels and implement risk mitigation strategies. By knowing the expected maximum drawdown with a high confidence level, managers can prepare contingency plans and set appropriate capital reserves.
          </Typography>
          
          <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>Client Communication</Typography>
          <Typography paragraph>
            Financial advisors can use DaR to set realistic expectations with clients about potential losses during market downturns. This helps prevent panic selling during temporary market declines by framing drawdowns as anticipated events within the expected risk range.
          </Typography>
          
          <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>Performance Evaluation</Typography>
          <Typography paragraph>
            DaR can be used alongside other metrics like Calmar Ratio (return divided by maximum drawdown) to evaluate investment strategies and managers, focusing on downside risk rather than just return volatility.
          </Typography>
          
          <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>Strategy Comparison</Typography>
          <Typography paragraph>
            When comparing different investment strategies or portfolios, DaR provides insight into the worst-case scenarios that each might face, helping investors choose options that align with their risk tolerance.
          </Typography>
        </Paper>

                {/* Advantages and Limitations */}        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>          <Typography variant="h5" component="h2" gutterBottom>Advantages and Limitations</Typography>                    <Grid container spacing={3}>            <Grid item xs={12} md={6}>              <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>                <Typography variant="h6" gutterBottom color="primary">                  Advantages                </Typography>                <ul>                  <li>                    <Typography paragraph>                      <strong>Investor-Centric:</strong> DaR directly addresses one of the most emotionally and financially significant aspects of investing—substantial losses from peak values.                    </Typography>                  </li>                  <li>                    <Typography paragraph>                      <strong>Path Sensitive:</strong> Unlike point-in-time risk measures, DaR captures the sequence and cumulative effect of returns, which better reflects the investor experience.                    </Typography>                  </li>                  <li>                    <Typography paragraph>                      <strong>Intuitive Interpretation:</strong> Expressed as a percentage loss, DaR is easier for non-technical stakeholders to understand than abstract statistical measures.                    </Typography>                  </li>                  <li>                    <Typography paragraph>                      <strong>Comprehensive:</strong> DaR inherently accounts for serial correlation, fat tails, and other realistic market characteristics without requiring specific distribution assumptions.                    </Typography>                  </li>                </ul>              </Box>            </Grid>            <Grid item xs={12} md={6}>              <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>                <Typography variant="h6" gutterBottom color="error">                  Limitations                </Typography>                <ul>                  <li>                    <Typography paragraph>                      <strong>Data Requirements:</strong> Accurate DaR estimation requires substantial historical data or advanced simulation techniques to capture extreme events.                    </Typography>                  </li>                  <li>                    <Typography paragraph>                      <strong>Non-Coherence:</strong> Unlike CVaR, DaR is not mathematically a coherent risk measure, which means it doesn't always satisfy the sub-additivity property crucial for capturing diversification benefits.                    </Typography>                  </li>                  <li>                    <Typography paragraph>                      <strong>Computational Complexity:</strong> Calculating DaR, especially for complex portfolios or using Monte Carlo simulations, can be computationally intensive.                    </Typography>                  </li>                  <li>                    <Typography paragraph>                      <strong>Backward-Looking:</strong> Like many risk measures, DaR based on historical data may not fully capture future risks, particularly in changing market regimes.                    </Typography>                  </li>                </ul>              </Box>            </Grid>          </Grid>        </Paper>

        {/* Related Metrics */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>Related Metrics</Typography>
          <Typography paragraph>
            DaR is part of a family of drawdown-based risk measures:
          </Typography>
          <ul>
            <li>
              <Typography paragraph>
                <strong>Conditional Drawdown at Risk (CDaR):</strong> The expected drawdown when exceeding the DaR threshold, similar to how CVaR relates to VaR.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Average Drawdown:</strong> The mean of all drawdowns over a time period, providing a measure of typical drawdown severity.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Maximum Drawdown (MDD):</strong> The largest percentage drop from peak to trough in a portfolio's value over a specific time period.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Drawdown Duration:</strong> The time it takes for a portfolio to recover from a drawdown and reach a new high, measuring recovery speed.
              </Typography>
            </li>
          </ul>
        </Paper>

        {/* Related Topics */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Related Topics
          </Typography>
          <Grid container spacing={2}>
            <Grid item xs={12} sm={6} md={4}>
              <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>
                <Typography variant="h6" gutterBottom>
                  Conditional Drawdown at Risk (CDaR)
                </Typography>
                <Typography variant="body2" paragraph>
                  The expected value of drawdowns exceeding the DaR threshold, measuring the severity of tail drawdown events.
                </Typography>
                <Link href="/docs/cdar" passHref>
                  <Button variant="contained" color="primary">
                    Learn More
                  </Button>
                </Link>
              </Box>
            </Grid>

            <Grid item xs={12} sm={6} md={4}>
              <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>
                <Typography variant="h6" gutterBottom>
                  Maximum Drawdown
                </Typography>
                <Typography variant="body2" paragraph>
                  The largest peak-to-trough decline in portfolio value, measuring worst historical loss from a previous peak.
                </Typography>
                <Link href="/docs/maximum-drawdown" passHref>
                  <Button variant="contained" color="primary">
                    Learn More
                  </Button>
                </Link>
              </Box>
            </Grid>

            <Grid item xs={12} sm={6} md={4}>
              <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>
                <Typography variant="h6" gutterBottom>
                  Conditional Value at Risk (CVaR)
                </Typography>
                <Typography variant="body2" paragraph>
                  A risk measure that quantifies the expected loss in the worst scenarios beyond the VaR threshold.
                </Typography>
                <Link href="/docs/conditional-value-at-risk" passHref>
                  <Button variant="contained" color="primary">
                    Learn More
                  </Button>
                </Link>
              </Box>
            </Grid>
          </Grid>
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
                Zabarankin, M., Pavlikov, K., & Uryasev, S. (2014). "Capital Asset Pricing Model (CAPM) with Drawdown Measure." European Journal of Operational Research, 234(2), 508-517.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                Carr, P., Zhang, H., & Hadjiliadis, O. (2011). "Maximum Drawdown Insurance." International Journal of Theoretical and Applied Finance, 14(08), 1195-1230.
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

export default DrawdownAtRiskPage; 