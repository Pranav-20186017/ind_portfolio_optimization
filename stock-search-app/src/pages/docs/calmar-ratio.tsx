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

const CalmarRatioPage: React.FC = () => {
  return (
    <>
      <Head>
        <title>Calmar Ratio for Indian Portfolios | QuantPort India Docs</title>
        <meta
          name="description"
          content="Learn about the Calmar Ratio for evaluating Indian equity portfolios. Measure risk-adjusted returns of NSE/BSE investments by comparing annualized returns to maximum drawdowns."
        />
        <meta property="og:title" content="Calmar Ratio for Indian Portfolios | QuantPort India Docs" />
        <meta property="og:description" content="Learn about the Calmar Ratio for evaluating Indian equity portfolios. Measure risk-adjusted returns of NSE/BSE investments by comparing annualized returns to maximum drawdowns." />
        <meta property="og:type" content="article" />
        <meta property="og:url" content="https://indportfoliooptimization.vercel.app/docs/calmar-ratio" />
      </Head>

      <Container maxWidth="lg" sx={{ py: 4 }}>
        {/* Navigation */}
        <Box sx={{ mb: 4, display: 'flex', gap: 2 }}>
          <Link href="/docs" passHref>
            <Button variant="outlined" color="primary">← Back to Docs</Button>
          </Link>
          <Link href="/" passHref>
            <Button variant="outlined" color="secondary">← Back to Portfolio Optimizer</Button>
          </Link>
        </Box>

        {/* Title */}
        <Box sx={{ mb: 6, textAlign: 'center' }}>
          <Typography variant="h3" component="h1" gutterBottom>
            Calmar Ratio
          </Typography>
          <Typography variant="subtitle1" color="text.secondary">
            Measuring risk-adjusted performance through maximum drawdown
          </Typography>
        </Box>

        {/* Overview */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>Overview</Typography>
          <Typography paragraph>
            The <strong>Calmar Ratio</strong> is a performance measurement developed by Terry W. Young in 1991 that evaluates 
            risk-adjusted returns by comparing the average annual compound rate of return to the maximum drawdown risk over a 
            specified time period. Named as an acronym for "California Managed Account Reports," it has become a staple metric 
            for evaluating hedge funds, managed futures accounts, and other alternative investments.
          </Typography>
          <Typography paragraph>
            The Calmar Ratio is particularly valuable for investors who are concerned about significant capital losses, as it directly 
            incorporates the worst historical drawdown experience rather than using standard deviation or other volatility 
            measures that may underweight extreme market events. Typically calculated using a three-year timeframe, the 
            ratio provides a straightforward assessment of return per unit of drawdown risk.
          </Typography>
        </Paper>

        {/* Intuitive Explanation */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>Intuitive Explanation</Typography>
          <Typography paragraph>
            Think of the Calmar Ratio as a measure of "reward for pain tolerance." It answers the question: "How much annual return 
            am I earning for each percentage point of my worst historical decline?"
          </Typography>
          <Typography paragraph>
            While the Sharpe ratio uses standard deviation to measure risk (which treats upside and downside volatility equally), 
            the Calmar ratio focuses exclusively on the downside—specifically, the worst peak-to-trough drop you would have 
            experienced as an investor. This approach recognizes that investors are typically more concerned about significant 
            losses than they are about return variability in general.
          </Typography>
          <Box sx={{ bgcolor: 'background.paper', p: 2, borderRadius: 1, border: '1px solid #e0e0e0', mt: 2 }}>
            <Typography variant="body1">
              <strong>Mountain climbing analogy:</strong> Imagine two mountain trails leading to the same peak (same final return). 
              The first trail has many small ups and downs but never drops too far below the previous high point. The second trail 
              has a massive descent in the middle before climbing to the peak. The Calmar ratio would favor the first path, recognizing 
              that most hikers (investors) prefer routes without stomach-dropping descents, even if the average steepness (volatility) 
              might be similar.
            </Typography>
          </Box>
        </Paper>

        {/* Mathematical Explanation */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>Detailed Mathematical Explanation</Typography>

          <Typography paragraph>
            The Calmar Ratio is calculated by dividing the average annual compound rate of return by the maximum drawdown 
            over the specified time period (typically 36 months):
          </Typography>

          <Box sx={{ p: 3, bgcolor: 'background.paper', borderRadius: 1, border: '1px solid #e0e0e0', my: 3 }}>
            <Typography variant="subtitle1" gutterBottom><strong>Calmar Ratio Formula</strong></Typography>
            <Equation math="\text{Calmar Ratio} = \frac{\text{Annualized Return}}{\text{Maximum Drawdown}}" />
          </Box>

          <Typography variant="h6" gutterBottom>Components:</Typography>
          
          <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>1. Annualized Return</Typography>
          <Typography paragraph>
            The annualized return is typically calculated as the compound annual growth rate (CAGR) over the evaluation period:
          </Typography>
          
          <Equation math="\text{Annualized Return} = \left(\frac{\text{Ending Value}}{\text{Beginning Value}}\right)^{\frac{1}{\text{Years}}} - 1" />

          <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>2. Maximum Drawdown</Typography>
          <Typography paragraph>
            The maximum drawdown represents the largest percentage drop from a peak to a subsequent trough during the measured period:
          </Typography>

          <Equation math="\text{Maximum Drawdown} = \max_{\forall t \in T} \left(\frac{\text{Peak Value prior to }t - \text{Value at }t}{\text{Peak Value prior to }t}\right)" />

          <Typography paragraph>
            Mathematically, if we define the drawdown at time t as:
          </Typography>

          <Equation math="\text{Drawdown}_t = \frac{\max_{s \in [0,t]} P_s - P_t}{\max_{s \in [0,t]} P_s}" />

          <Typography paragraph>
            where <InlineMath math="P_t" /> is the portfolio value at time t, then the maximum drawdown is:
          </Typography>

          <Equation math="\text{Maximum Drawdown} = \max_{t \in T} \text{Drawdown}_t" />
          
          <Typography paragraph>
            The maximum drawdown is always expressed as a positive percentage, even though it represents a loss. 
            A maximum drawdown of 0.20 (or 20%) means that the portfolio experienced a 20% decline from its previous peak at some point.
          </Typography>
        </Paper>

        {/* Implementation Details */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>Implementation in Our Service</Typography>
          <Typography paragraph>
            Our portfolio analyzer calculates the Calmar Ratio through the following steps:
          </Typography>
          <ul>
            <li>
              <Typography paragraph>
                <strong>Return Calculation:</strong> We compute the annualized return over the specified period (default is 36 months) using the CAGR formula.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Drawdown Series:</strong> We generate the complete drawdown series by tracking the percentage decline from the running maximum portfolio value.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Maximum Drawdown:</strong> We identify the largest value in the drawdown series within the evaluation period.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Ratio Computation:</strong> We divide the annualized return by the maximum drawdown to derive the Calmar Ratio.
              </Typography>
            </li>
          </ul>
          <Typography paragraph>
            By default, we use a 36-month (3-year) period for the Calmar Ratio calculation, though this timeframe can be adjusted based on user preference. For portfolios with shorter histories, we calculate the ratio using all available data but clearly indicate when the measurement period is less than the standard 36 months.
          </Typography>
        </Paper>

        {/* Worked Example */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>Worked Example</Typography>
          <Typography paragraph>
            Let's calculate the Calmar Ratio for a hypothetical portfolio over a 3-year period:
          </Typography>

          <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>Step 1: Calculate the annualized return</Typography>
          <Typography paragraph>
            Initial value: $100,000<br />
            Final value after 3 years: $130,000
          </Typography>
          <Typography component="div" paragraph>
            <Equation math="\text{Annualized Return} = \left(\frac{\$130,000}{\$100,000}\right)^{\frac{1}{3}} - 1 = 1.3^{0.333} - 1 \approx 0.0914 \text{ or } 9.14\%" />
          </Typography>

          <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>Step 2: Identify the maximum drawdown</Typography>
          <Typography paragraph>
            In this example, let's assume our portfolio experienced the following drawdowns over the 3-year period:
          </Typography>
          <ul>
            <li>
              <Typography paragraph>Year 1, Month 5: -8% drawdown</Typography>
            </li>
            <li>
              <Typography paragraph>Year 2, Month 2: -15% drawdown</Typography>
            </li>
            <li>
              <Typography paragraph>Year 3, Month 7: -12% drawdown</Typography>
            </li>
          </ul>
          <Typography paragraph>
            The maximum drawdown is therefore 15%.
          </Typography>

          <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>Step 3: Compute the Calmar Ratio</Typography>
          <Typography component="div" paragraph>
            <Equation math="\text{Calmar Ratio} = \frac{0.0914}{0.15} \approx 0.61" />
          </Typography>

          <Typography paragraph>
            A Calmar Ratio of 0.61 indicates that the portfolio generates 0.61 units of annualized return for each unit of maximum drawdown risk. 
            In other words, investors earned 9.14% annual returns while enduring a worst-case scenario of a 15% decline from peak to trough.
          </Typography>

          <Typography paragraph>
            How should we interpret this value? Generally:
          </Typography>
          <ul>
            <li>
              <Typography paragraph><strong>Calmar Ratio {'>'}  1:</strong> Considered good (annual return exceeds worst drawdown)</Typography>
            </li>
            <li>
              <Typography paragraph><strong>Calmar Ratio {'>'}  2:</strong> Considered excellent (annual return is double the worst drawdown)</Typography>
            </li>
            <li>
              <Typography paragraph><strong>Calmar Ratio {'<'}  0.5:</strong> May indicate poor risk-adjusted performance (annual return is less than half the worst drawdown)</Typography>
            </li>
          </ul>
        </Paper>

        {/* Applications */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>Practical Applications</Typography>
          <Typography paragraph>
            The Calmar Ratio is particularly useful in several investment contexts:
          </Typography>
          <ul>
            <li>
              <Typography paragraph>
                <strong>Hedge Fund Evaluation:</strong> Many hedge funds and alternative investments specifically report Calmar Ratios since these vehicles often employ strategies that may have non-normal return distributions or significant tail risks.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Comparing Tactical Strategies:</strong> When evaluating active or tactical trading strategies that aim to avoid market downturns, the Calmar Ratio can provide more relevant information than traditional volatility-based metrics.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Retirement Planning:</strong> For investors approaching or in retirement, the Calmar Ratio can be particularly relevant as large drawdowns during this period can have severe consequences (sometimes called "sequence of returns risk").
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Risk Budgeting:</strong> Portfolio managers can use Calmar Ratios to allocate capital to strategies based on how efficiently they use their "drawdown budget."
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Manager Selection:</strong> When comparing managers within the same investment category, the Calmar Ratio can help identify which ones deliver the best returns relative to their worst historical drawdown.
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
                      <strong>Intuitively meaningful:</strong> Directly connects returns to the worst pain point an investor would have experienced, which aligns with how many investors actually perceive risk.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Focus on extreme events:</strong> Captures tail risk and worst-case scenarios that might be underrepresented in standard deviation-based measures like the Sharpe ratio.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Simplicity:</strong> Easy to calculate and interpret without complex statistical assumptions about return distributions.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Penalizes volatility clustering:</strong> Strategies that experience concentrated losses are heavily penalized, even if their overall volatility appears manageable.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Time awareness:</strong> By using drawdown, the metric naturally captures the duration and path dependency of losses, not just their magnitude.
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
                      <strong>Single point sensitivity:</strong> Relies on a single worst-case event that may be an anomaly or unlikely to repeat, potentially overweighting one-time market shocks.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Time period dependence:</strong> Results can vary significantly based on the chosen evaluation period, which may not include major market downturns if too short.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Backward-looking:</strong> Like all historical performance metrics, assumes that past drawdown patterns are representative of future risks.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Ignores recovery time:</strong> Two investments could have identical maximum drawdowns but vastly different recovery periods, which the Calmar Ratio doesn't distinguish.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>No statistical confidence:</strong> Unlike some other metrics, there is no established framework for determining statistical significance or confidence intervals for the Calmar Ratio.
                    </Typography>
                  </li>
                </ul>
              </Box>
            </Grid>
          </Grid>
        </Paper>

        {/* Sterling Ratio Comparison */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>Comparison with Sterling Ratio</Typography>
          <Typography paragraph>
            The Calmar Ratio is often compared to the Sterling Ratio, another drawdown-based performance metric. The key difference is that the Sterling Ratio uses the average of the annual maximum drawdowns (often minus 10%) rather than the single worst drawdown:
          </Typography>
          
          <Equation math="\text{Sterling Ratio} = \frac{\text{Annualized Return}}{\text{Average Annual Max Drawdown} - 10\%}" />
          
          <Typography paragraph>
            The Sterling Ratio tends to be less sensitive to a single extreme drawdown event but still captures the pattern of significant losses. Some practitioners prefer the Sterling Ratio when evaluating investments with longer track records because it provides a more comprehensive view of drawdown history rather than focusing only on the single worst event.
          </Typography>
        </Paper>

        {/* References */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>References</Typography>
          <ul>
            <li>
              <Typography paragraph><strong>Young, T.W. (1991)</strong>. "Calmar Ratio: A Smoother Tool." <em>Futures Magazine</em>.</Typography>
            </li>
            <li>
              <Typography paragraph><strong>Magdon-Ismail, M., & Atiya, A. (2004)</strong>. "Maximum drawdown." <em>Risk Magazine</em>, 17(10), 99-102.</Typography>
            </li>
            <li>
              <Typography paragraph><strong>Schuhmacher, F., & Eling, M. (2011)</strong>. "Sufficient conditions for expected utility to imply drawdown-based performance rankings." <em>Journal of Banking & Finance</em>, 35(9), 2311-2318.</Typography>
            </li>
            <li>
              <Typography paragraph><strong>Bacon, C. R. (2013)</strong>. <em>Practical Risk-Adjusted Performance Measurement</em>. Wiley.</Typography>
            </li>
            <li>
              <Typography paragraph><strong>Caporin, M., & Lisi, F. (2011)</strong>. "Comparing and selecting performance measures using rank correlations." <em>Economics: The Open-Access, Open-Assessment E-Journal</em>, 5(2011-10), 1-34.</Typography>
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
                <Typography variant="body2" paragraph>The classic risk-adjusted return measure using standard deviation as the risk metric.</Typography>
                <Link href="/docs/sharpe-ratio" passHref>
                  <Button variant="contained" color="primary">Learn More</Button>
                </Link>
              </Box>
            </Grid>
            <Grid item xs={12} sm={6} md={4}>
              <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>
                <Typography variant="h6" gutterBottom>Sortino Ratio</Typography>
                <Typography variant="body2" paragraph>A risk-adjusted measure focusing only on downside deviation below a minimum acceptable return.</Typography>
                <Link href="/docs/sortino-ratio" passHref>
                  <Button variant="contained" color="primary">Learn More</Button>
                </Link>
              </Box>
            </Grid>
            <Grid item xs={12} sm={6} md={4}>
              <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>
                <Typography variant="h6" gutterBottom>Omega Ratio</Typography>
                <Typography variant="body2" paragraph>A performance measure evaluating the probability-weighted ratio of gains versus losses for a threshold return.</Typography>
                <Link href="/docs/omega-ratio" passHref>
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

export default CalmarRatioPage; 