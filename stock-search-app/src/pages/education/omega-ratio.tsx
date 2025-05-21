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

const OmegaRatioPage: React.FC = () => {
  return (
    <>
      <Head>
        <title>Omega Ratio (Ω) | Portfolio Optimization</title>
        <meta
          name="description"
          content="Learn about the Omega Ratio (Ω), a performance measure that evaluates the probability-weighted ratio of gains versus losses for a threshold return."
        />
      </Head>

      <Container maxWidth="lg" sx={{ py: 4 }}>
        {/* Navigation */}
        <Box sx={{ mb: 4, display: 'flex', gap: 2 }}>
          <Link href="/education" passHref>
            <Button variant="outlined" color="primary">← Back to Education</Button>
          </Link>
          <Link href="/" passHref>
            <Button variant="outlined" color="secondary">← Back to Portfolio Optimizer</Button>
          </Link>
        </Box>

        {/* Title */}
        <Box sx={{ mb: 6, textAlign: 'center' }}>
          <Typography variant="h3" component="h1" gutterBottom>
            Omega Ratio (Ω)
          </Typography>
          <Typography variant="subtitle1" color="text.secondary">
            A comprehensive performance measure that captures the entire return distribution
          </Typography>
        </Box>

        {/* Overview */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>Overview</Typography>
          <Typography paragraph>
            The <strong>Omega Ratio (Ω)</strong> is a comprehensive performance measure introduced by 
            <em> Keating and Shadwick</em> in 2002 that evaluates the probability-weighted ratio of gains versus losses 
            relative to a threshold return. Unlike traditional metrics such as the Sharpe ratio that primarily focus on mean and variance, 
            the Omega ratio incorporates the entire return distribution, making it particularly valuable for evaluating 
            investments with non-normal return distributions.
          </Typography>
          <Typography paragraph>
            By considering the entire probability distribution of returns, the Omega ratio accounts for 
            asymmetry, fat tails, and other higher moments that are often ignored by traditional performance metrics.
            This makes it especially useful for evaluating alternative investments, hedge funds, and complex strategies
            whose returns often exhibit skewness and kurtosis.
          </Typography>
        </Paper>

        {/* Intuitive Explanation */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>Intuitive Explanation</Typography>
          <Typography paragraph>
            Think of the Omega ratio as a comprehensive "odds calculator" for your investments. It answers the question: 
            "What are the odds of achieving returns above my minimum acceptable threshold versus falling below it?"
          </Typography>
          <Typography paragraph>
            Imagine you're deciding between two investment strategies. Both have the same average return and standard deviation, 
            but one has occasional large gains while the other has more consistent moderate returns. Traditional metrics like the 
            Sharpe ratio would rate them similarly, but the Omega ratio would highlight the difference by evaluating the entire 
            shape of their return distributions.
          </Typography>
          <Box sx={{ bgcolor: 'background.paper', p: 2, borderRadius: 1, border: '1px solid #e0e0e0', mt: 2 }}>
            <Typography variant="body1">
              <strong>Casino analogy:</strong> Consider two different slot machines. Both have the same average payout over time,
              but one pays small amounts frequently while the other rarely pays but gives large jackpots when it does. 
              The Omega ratio would help determine which machine is more likely to keep you above your "break-even" threshold, 
              accounting for both the frequency and magnitude of all possible outcomes.
            </Typography>
          </Box>
        </Paper>

        {/* Mathematical Explanation */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>Detailed Mathematical Explanation</Typography>

          <Typography paragraph>
            Mathematically, the Omega ratio is defined as the ratio of the probability-weighted gains to the 
            probability-weighted losses, relative to a threshold return <InlineMath math="\tau" />:
          </Typography>

          <Box sx={{ p: 3, bgcolor: 'background.paper', borderRadius: 1, border: '1px solid #e0e0e0', my: 3 }}>
            <Typography variant="subtitle1" gutterBottom><strong>Omega Ratio Formula</strong></Typography>
            <Equation math="\Omega(r, \tau) = \frac{\int_{\tau}^{\infty} [1 - F(r)] \, dr}{\int_{-\infty}^{\tau} F(r) \, dr}" />
            <Typography variant="body2">
              where <InlineMath math="F(r)" /> is the cumulative distribution function (CDF) of returns <InlineMath math="r" />, and
              <InlineMath math="\tau" /> is the threshold return.
            </Typography>
          </Box>

          <Typography paragraph>
            An equivalent representation using the partial expectation functions is:
          </Typography>

          <Equation math="\Omega(r, \tau) = \frac{E[(r-\tau)^+]}{E[(\tau-r)^+]}" />

          <Typography paragraph>
            where <InlineMath math="(r-\tau)^+" /> represents the positive part, <InlineMath math="\max(r-\tau, 0)" />.
          </Typography>

          <Typography variant="h6" gutterBottom sx={{ mt: 3 }}>Practical Calculation</Typography>
          <Typography paragraph>
            In practice, the Omega ratio is often computed from a set of historical returns by:
          </Typography>

          <ul>
            <li>
              <Typography paragraph>
                Sorting returns in ascending order.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                Constructing the empirical distribution function.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                Computing the areas above and below the threshold.
              </Typography>
            </li>
          </ul>

          <Typography paragraph>
            For discrete returns <InlineMath math="r_1, r_2, \ldots, r_n" />, the Omega ratio calculation simplifies to:
          </Typography>

          <Equation math="\Omega(\tau) = \frac{\sum_{i=1}^{n} \max(r_i - \tau, 0)}{\sum_{i=1}^{n} \max(\tau - r_i, 0)}" />
          
          <Typography paragraph>
            This can be interpreted as the ratio of the average gain above the threshold to the average loss below the threshold.
          </Typography>
        </Paper>

        {/* Properties */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>Key Properties</Typography>
          <Grid container spacing={2}>
            <Grid item xs={12} md={6}>
              <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>
                <Typography variant="h6" gutterBottom>Threshold Dependence</Typography>
                <Typography paragraph>
                  The Omega ratio equals 1 when the threshold equals the mean return. As the threshold increases, 
                  the Omega ratio decreases monotonically. This property makes it useful for comparing investments across different thresholds.
                </Typography>
                <Equation math="\Omega(\mu) = 1 \quad \text{and} \quad \frac{d\Omega(\tau)}{d\tau} < 0 \quad \text{for} \quad \tau > \mu" />
              </Box>
            </Grid>
            <Grid item xs={12} md={6}>
              <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>
                <Typography variant="h6" gutterBottom>Higher Moment Sensitivity</Typography>
                <Typography paragraph>
                  The Omega ratio captures the effects of skewness, kurtosis, and all higher moments of the return distribution.
                  This makes it particularly valuable for non-normal returns where traditional metrics can be misleading.
                </Typography>
                <Typography>
                  For normal distributions with the same mean and variance, two investments would have identical Omega ratios.
                  However, for real-world return distributions, the Omega ratio can reveal differences that the Sharpe ratio misses.
                </Typography>
              </Box>
            </Grid>
          </Grid>
        </Paper>

        {/* Implementation Details */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>Implementation in Our Service</Typography>
          <Typography paragraph>
            Our portfolio analyzer implements the Omega ratio calculation through the following steps:
          </Typography>
          <ul>
            <li>
              <Typography paragraph>
                <strong>Threshold Setting:</strong> By default, we use the risk-free rate as the threshold, but users can customize this value based on their specific requirements.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Return Partitioning:</strong> Historical returns are partitioned into gains (returns above threshold) and losses (returns below threshold).
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Ratio Calculation:</strong> The sum of excess returns above the threshold is divided by the sum of shortfalls below the threshold.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Visualization:</strong> We provide an Omega function graph that plots the Omega ratio across different threshold values, giving a comprehensive view of performance across various return requirements.
              </Typography>
            </li>
          </ul>
        </Paper>

        {/* Worked Example */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>Worked Example</Typography>
          <Typography paragraph>
            Consider a portfolio with the following monthly returns over a year:
          </Typography>
          <Typography paragraph>
            2.1%, 1.5%, -0.8%, 3.2%, -1.7%, 0.6%, 1.9%, -0.3%, 2.5%, -2.1%, 1.1%, 2.8%
          </Typography>
          <Typography paragraph>
            Let's calculate the Omega ratio with a threshold of 0.5% (0.005):
          </Typography>
          <Typography component="div" paragraph>
            <strong>Step 1:</strong> Identify returns above the threshold: 2.1%, 1.5%, 3.2%, 0.6%, 1.9%, 2.5%, 1.1%, 2.8%
          </Typography>
          <Typography component="div" paragraph>
            <strong>Step 2:</strong> Identify returns below the threshold: -0.8%, -1.7%, -0.3%, -2.1%
          </Typography>
          <Typography component="div" paragraph>
            <strong>Step 3:</strong> Calculate the excess above threshold:
            <Equation math="\text{Excess} = (0.021 - 0.005) + (0.015 - 0.005) + ... + (0.028 - 0.005) = 0.118" />
          </Typography>
          <Typography component="div" paragraph>
            <strong>Step 4:</strong> Calculate the shortfall below threshold:
            <Equation math="\text{Shortfall} = (0.005 + 0.008) + (0.005 + 0.017) + (0.005 + 0.003) + (0.005 + 0.021) = 0.069" />
          </Typography>
          <Typography component="div" paragraph>
            <strong>Step 5:</strong> Compute the Omega ratio:
            <Equation math="\Omega(0.005) = \frac{0.118}{0.069} \approx 1.71" />
          </Typography>
          <Typography paragraph>
            An Omega ratio of 1.71 suggests that the probability-weighted upside potential is 1.71 times greater than the downside risk, 
            relative to the threshold of 0.5%. This indicates a favorable risk-reward profile at this threshold.
          </Typography>
        </Paper>

        {/* Practical Applications */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>Practical Applications</Typography>
          <Typography paragraph>
            The Omega ratio finds valuable applications in various investment contexts:
          </Typography>
          <ul>
            <li>
              <Typography paragraph>
                <strong>Portfolio Selection:</strong> Choosing between portfolios by comparing their Omega ratios at various threshold levels.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Hedge Fund Evaluation:</strong> Assessing hedge funds and alternative investments with non-normal return distributions.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Risk Budgeting:</strong> Allocating capital among strategies based on their Omega profiles for different thresholds.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Performance Attribution:</strong> Evaluating how different portfolio components contribute to overall performance relative to a required return.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Investor-Specific Assessment:</strong> Tailoring the evaluation of investments to an investor's specific minimum acceptable return.
              </Typography>
            </li>
          </ul>
        </Paper>

        {/* Advantages and Limitations */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>
            Advantages and Limitations
          </Typography>
          
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>
                <Typography variant="h6" gutterBottom color="primary">
                  Advantages
                </Typography>
                <ul>
                  <li>
                    <Typography paragraph>
                      <strong>Complete distribution information:</strong> Considers the entire return distribution instead of just the first two moments (mean and variance).
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Threshold flexibility:</strong> Allows for customization of the threshold based on investor-specific requirements or risk appetite.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Higher moment sensitivity:</strong> Accounts for skewness, kurtosis, and other higher moments that are important for non-normal distributions.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Intuitive interpretation:</strong> Can be understood as the odds of exceeding the threshold, providing a natural risk-reward measure.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Theoretical soundness:</strong> Consistent with expected utility theory and stochastic dominance principles from financial economics.
                    </Typography>
                  </li>
                </ul>
              </Box>
            </Grid>
            <Grid item xs={12} md={6}>
              <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>
                <Typography variant="h6" gutterBottom color="error">
                  Limitations
                </Typography>
                <ul>
                  <li>
                    <Typography paragraph>
                      <strong>Computational complexity:</strong> More computationally intensive than simpler metrics like the Sharpe ratio, especially for large datasets.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Threshold dependence:</strong> Results vary based on the chosen threshold, requiring analysis at multiple thresholds for a complete picture.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Data requirements:</strong> Needs substantial historical data to accurately estimate the return distribution, particularly the tails.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Stationarity assumption:</strong> Like many financial metrics, it assumes future returns will follow patterns similar to historical data.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Less established:</strong> Not as widely recognized or reported as traditional metrics like Sharpe or Sortino ratios in the investment industry.
                    </Typography>
                  </li>
                </ul>
              </Box>
            </Grid>
          </Grid>
        </Paper>

        {/* References */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>References</Typography>
          <ul>
            <li>
              <Typography paragraph><strong>Keating, C., & Shadwick, W. F. (2002)</strong>. "A Universal Performance Measure." <em>Journal of Performance Measurement</em>, 6(3), 59-84.</Typography>
            </li>
            <li>
              <Typography paragraph><strong>Kazemi, H., Schneeweis, T., & Gupta, R. (2004)</strong>. "Omega as a Performance Measure." <em>Journal of Performance Measurement</em>, 8(3), 16-25.</Typography>
            </li>
            <li>
              <Typography paragraph><strong>Favre-Bulle, A., & Pache, S. (2003)</strong>. "The Omega Measure: Hedge Fund Portfolio Optimization." <em>MBF Master's Thesis, University of Lausanne</em>.</Typography>
            </li>
            <li>
              <Typography paragraph><strong>Mausser, H., Saunders, D., & Seco, L. (2006)</strong>. "Optimizing Omega." <em>Risk</em>, 19(11), 88-92.</Typography>
            </li>
            <li>
              <Typography paragraph><strong>Bacmann, J. F., & Scholz, S. (2003)</strong>. "Alternative Performance Measures for Hedge Funds." <em>AIMA Journal</em>, 1(1), 1-9.</Typography>
            </li>
          </ul>
        </Paper>

        {/* Related Topics */}
        <Paper elevation={2} sx={{ p: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>Related Topics</Typography>
          <Grid container spacing={2}>
            <Grid item xs={12} sm={6} md={4}>
              <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>
                <Typography variant="h6" gutterBottom>Sharpe Ratio</Typography>
                <Typography variant="body2" paragraph>The traditional risk-adjusted return measure using standard deviation as the risk metric.</Typography>
                <Link href="/education/sharpe-ratio" passHref>
                  <Button variant="contained" color="primary">Learn More</Button>
                </Link>
              </Box>
            </Grid>
            <Grid item xs={12} sm={6} md={4}>
              <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>
                <Typography variant="h6" gutterBottom>Sortino Ratio</Typography>
                <Typography variant="body2" paragraph>A risk-adjusted measure focusing only on downside deviation below a minimum acceptable return.</Typography>
                <Link href="/education/sortino-ratio" passHref>
                  <Button variant="contained" color="primary">Learn More</Button>
                </Link>
              </Box>
            </Grid>
            <Grid item xs={12} sm={6} md={4}>
              <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>
                <Typography variant="h6" gutterBottom>Entropy</Typography>
                <Typography variant="body2" paragraph>A measure of uncertainty or randomness in portfolio returns, indicating the level of unpredictability.</Typography>
                <Link href="/education/entropy" passHref>
                  <Button variant="contained" color="primary">Learn More</Button>
                </Link>
              </Box>
            </Grid>
          </Grid>
        </Paper>
      </Container>
    </>
  );
};

// Static generation hook (Next.js)
export const getStaticProps = async () => {
  return { props: {} };
};

export default OmegaRatioPage; 