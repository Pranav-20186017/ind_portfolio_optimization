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

const UlcerIndexPage: React.FC = () => {
  return (
    <>
      <Head>
        <title>Ulcer Index | Portfolio Optimization</title>
        <meta
          name="description"
          content="Learn about the Ulcer Index, a volatility measure that captures the depth and duration of drawdowns, focusing on downside movement."
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
            Ulcer Index
          </Typography>
          <Typography variant="subtitle1" color="text.secondary">
            A volatility measure that focuses on the depth and duration of drawdowns
          </Typography>
        </Box>

        {/* Overview */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>Overview</Typography>
          <Typography paragraph>
            The <strong>Ulcer Index</strong> is a specialized volatility metric developed by Peter Martin and Byron McCann in 1987. 
            Unlike traditional volatility measures like standard deviation that treat upside and downside movements equally, 
            the Ulcer Index focuses exclusively on downside risk by measuring the depth and duration of price declines 
            (drawdowns) from previous highs. The name "Ulcer Index" aptly reflects the anxiety investors experience 
            during these periods of portfolio decline.
          </Typography>
          <Typography paragraph>
            This metric is particularly valuable for risk-averse investors who are more concerned with avoiding significant 
            drawdowns than with overall volatility. The Ulcer Index provides a more nuanced view of downside risk by penalizing 
            deeper and longer-lasting drawdowns more heavily, aligning with the psychological reality that investors tend to 
            feel progressively more pain the longer their portfolios remain below previous peak values.
          </Typography>
        </Paper>

        {/* Intuitive Explanation */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>Intuitive Explanation</Typography>
          <Typography paragraph>
            Think of the Ulcer Index as a "pain meter" for investments. While standard deviation measures all price movements 
            (both up and down) equally, the Ulcer Index only cares about how far an investment has fallen from its peak and 
            how long it stays down—the two factors that cause investors the most distress.
          </Typography>
          <Typography paragraph>
            Consider two portfolios with the same average returns and standard deviation over a year. Portfolio A experiences 
            a single sharp 15% drop but quickly recovers, while Portfolio B suffers a 15% decline that persists for several months 
            before recovering. Traditional volatility measures would rate these portfolios similarly, but the Ulcer Index would 
            assign a higher "pain score" to Portfolio B because of the extended duration of its drawdown.
          </Typography>
          <Box sx={{ bgcolor: 'background.paper', p: 2, borderRadius: 1, border: '1px solid #e0e0e0', mt: 2 }}>
            <Typography variant="body1">
              <strong>Health analogy:</strong> Standard deviation is like measuring someone's average body temperature fluctuations 
              throughout the day. The Ulcer Index is more like tracking how far below normal a patient's temperature drops during 
              an illness and how many days they remain sick. It's not just that the temperature is abnormal (volatility), 
              but how severe the illness is and how long it lasts (drawdown depth and duration) that really matters for patient comfort.
            </Typography>
          </Box>
        </Paper>

        {/* Mathematical Explanation */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>Detailed Mathematical Explanation</Typography>

          <Typography paragraph>
            The Ulcer Index is calculated by taking the square root of the mean of the squared percentage drawdowns from historical peak values:
          </Typography>

          <Box sx={{ p: 3, bgcolor: 'background.paper', borderRadius: 1, border: '1px solid #e0e0e0', my: 3 }}>
            <Typography variant="subtitle1" gutterBottom><strong>Ulcer Index Formula</strong></Typography>
            <Equation math="\text{UI} = \sqrt{\frac{1}{n}\sum_{i=1}^{n} R_i^2}" />
            <Typography variant="body2">
              where <InlineMath math="R_i" /> represents the percentage drawdown at time period <InlineMath math="i" /> and <InlineMath math="n" /> is the number of time periods.
            </Typography>
          </Box>

          <Typography variant="h6" gutterBottom sx={{ mt: 3 }}>Computing the Percentage Drawdown</Typography>
          <Typography paragraph>
            For each time period <InlineMath math="i" />, the percentage drawdown <InlineMath math="R_i" /> is calculated as:
          </Typography>

          <Equation math="R_i = \frac{P_i - P_{max}}{P_{max}} \times 100\%" />

          <Typography paragraph>
            where:
          </Typography>
          <ul>
            <li>
              <Typography paragraph>
                <InlineMath math="P_i" /> is the price (or portfolio value) at time period <InlineMath math="i" />
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <InlineMath math="P_{max}" /> is the maximum price (or highest portfolio value) up to time period <InlineMath math="i" />
              </Typography>
            </li>
          </ul>

          <Typography paragraph>
            The key properties of this calculation are:
          </Typography>
          <ul>
            <li>
              <Typography paragraph>
                <InlineMath math="R_i = 0" /> when price is at a new peak (no drawdown)
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <InlineMath math="R_i" /> is always negative during drawdowns (but the squared value makes it positive in the calculation)
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                Squaring the drawdowns gives greater weight to larger percentage drops
              </Typography>
            </li>
          </ul>

          <Typography variant="h6" gutterBottom sx={{ mt: 3 }}>Ulcer Performance Index (UPI)</Typography>
          <Typography paragraph>
            A related measure is the Ulcer Performance Index (UPI), also known as the Martin Ratio, which is calculated by dividing excess returns by the Ulcer Index:
          </Typography>

          <Equation math="\text{UPI} = \frac{R_p - R_f}{\text{UI}}" />

          <Typography paragraph>
            where <InlineMath math="R_p" /> is the portfolio return and <InlineMath math="R_f" /> is the risk-free rate. The UPI is analogous to the Sharpe ratio but uses the Ulcer Index as the risk measure instead of standard deviation.
          </Typography>
        </Paper>

        {/* Implementation Details */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>Implementation in Our Service</Typography>
          <Typography paragraph>
            Our portfolio analyzer calculates the Ulcer Index through the following process:
          </Typography>
          <ul>
            <li>
              <Typography paragraph>
                <strong>Price Series Analysis:</strong> We track the value of the portfolio over time, identifying the maximum value reached up to each point.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Drawdown Calculation:</strong> For each time period, we calculate the percentage decline from the highest peak to the current value.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Squaring Process:</strong> We square each percentage drawdown to penalize larger drops more heavily.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Averaging and Square Root:</strong> We take the mean of these squared drawdowns and then calculate the square root to produce the final Ulcer Index.
              </Typography>
            </li>
          </ul>
          <Typography paragraph>
            We also provide the Ulcer Performance Index (UPI) to give users a risk-adjusted performance measure that's particularly sensitive to drawdowns. For backtesting purposes, we allow users to view how the Ulcer Index evolves over time, helping identify periods of increased drawdown risk.
          </Typography>
        </Paper>

        {/* Worked Example */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>Worked Example</Typography>
          <Typography paragraph>
            Let's calculate the Ulcer Index for a portfolio with the following month-end values over a 10-month period:
          </Typography>
          <Typography paragraph>
            $100, $102, $104, $99, $101, $97, $95, $98, $102, $105
          </Typography>

          <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>Step 1: Identify the running maximum at each point</Typography>
          <Typography paragraph>
            Track the highest value achieved up to each time period:
          </Typography>
          <Typography paragraph>
            $100, $102, $104, $104, $104, $104, $104, $104, $104, $105
          </Typography>

          <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>Step 2: Calculate the percentage drawdowns</Typography>
          <Typography paragraph>
            For each month, calculate how far the portfolio is below its previous peak:
          </Typography>

          <ul>
            <li><Typography paragraph>Month 1: ($100 - $100)/$100 × 100% = 0% (no drawdown)</Typography></li>
            <li><Typography paragraph>Month 2: ($102 - $102)/$102 × 100% = 0% (no drawdown)</Typography></li>
            <li><Typography paragraph>Month 3: ($104 - $104)/$104 × 100% = 0% (no drawdown)</Typography></li>
            <li><Typography paragraph>Month 4: ($99 - $104)/$104 × 100% = -4.81%</Typography></li>
            <li><Typography paragraph>Month 5: ($101 - $104)/$104 × 100% = -2.88%</Typography></li>
            <li><Typography paragraph>Month 6: ($97 - $104)/$104 × 100% = -6.73%</Typography></li>
            <li><Typography paragraph>Month 7: ($95 - $104)/$104 × 100% = -8.65%</Typography></li>
            <li><Typography paragraph>Month 8: ($98 - $104)/$104 × 100% = -5.77%</Typography></li>
            <li><Typography paragraph>Month 9: ($102 - $104)/$104 × 100% = -1.92%</Typography></li>
            <li><Typography paragraph>Month 10: ($105 - $105)/$105 × 100% = 0% (new peak)</Typography></li>
          </ul>

          <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>Step 3: Square the percentage drawdowns</Typography>
          <Typography paragraph>
            0², 0², 0², 4.81², 2.88², 6.73², 8.65², 5.77², 1.92², 0² = 0, 0, 0, 23.14, 8.29, 45.29, 74.82, 33.29, 3.69, 0
          </Typography>

          <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>Step 4: Calculate the mean of squared drawdowns</Typography>
          <Typography component="div" paragraph>
            <Equation math="\text{Mean} = \frac{0 + 0 + 0 + 23.14 + 8.29 + 45.29 + 74.82 + 33.29 + 3.69 + 0}{10} = \frac{188.52}{10} = 18.85" />
          </Typography>

          <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>Step 5: Take the square root for the final Ulcer Index</Typography>
          <Typography component="div" paragraph>
            <Equation math="\text{UI} = \sqrt{18.85} \approx 4.34\%" />
          </Typography>

          <Typography paragraph>
            This Ulcer Index of 4.34% represents the severity of drawdowns experienced by the portfolio over the 10-month period. The higher the index, the more severe the combination of drawdown depth and duration.
          </Typography>

          <Typography paragraph>
            If the portfolio had an annualized return of 7% during this period and the risk-free rate was 2%, the Ulcer Performance Index (UPI) would be:
          </Typography>

          <Typography component="div" paragraph>
            <Equation math="\text{UPI} = \frac{7\% - 2\%}{4.34\%} \approx 1.15" />
          </Typography>

          <Typography paragraph>
            A UPI of 1.15 indicates the portfolio generated 1.15 units of excess return per unit of drawdown risk.
          </Typography>
        </Paper>

        {/* Applications */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>Practical Applications</Typography>
          <Typography paragraph>
            The Ulcer Index is particularly valuable in several investment contexts:
          </Typography>
          <ul>
            <li>
              <Typography paragraph>
                <strong>Retirement Portfolio Management:</strong> Retirees making regular withdrawals are especially vulnerable to sequence-of-returns risk. The Ulcer Index helps identify strategies that minimize drawdown depth and duration, critical for preserving longevity in retirement accounts.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Risk-Averse Client Portfolios:</strong> For investors with lower risk tolerance, focusing on minimizing the Ulcer Index rather than standard deviation can better align with their psychological comfort and prevent panic selling during market declines.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Tactical Asset Allocation:</strong> The Ulcer Index can be used to evaluate the effectiveness of tactical strategies designed to reduce drawdown risk, providing a more appropriate metric than standard deviation for these approaches.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Manager Evaluation:</strong> When comparing investment managers, particularly those with capital preservation mandates, the Ulcer Index and UPI provide more relevant performance metrics than conventional Sharpe ratios.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Portfolio Construction:</strong> Including the Ulcer Index as a minimization objective in portfolio optimization can help construct portfolios that specifically target drawdown reduction rather than just overall volatility reduction.
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
                      <strong>Downside focus:</strong> Exclusively measures downside risk, aligning with investors' primary concern of avoiding losses rather than reducing overall volatility.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Duration sensitivity:</strong> Captures the length of time investments remain underwater, acknowledging the psychological impact of extended drawdown periods.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Squared penalties:</strong> By squaring drawdowns, the index disproportionately penalizes larger drops, reflecting the non-linear nature of risk perception.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Path dependency:</strong> Accounts for the specific sequence of returns rather than just their overall distribution, acknowledging the importance of when losses occur.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Intuitive interpretation:</strong> Directly measures what causes investors the most pain—deep, prolonged declines from previous peaks.
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
                      <strong>Lookback limitations:</strong> The index depends heavily on the chosen time period; too short a period may miss significant historical drawdowns.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Peak-anchoring bias:</strong> Always references the previous peak, potentially overstating risk if that peak was an anomalous high point.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Frequency sensitivity:</strong> Results can vary significantly based on the frequency of data sampling (daily vs. weekly vs. monthly).
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Lesser adoption:</strong> Not as widely recognized or used as standard deviation or VaR, making comparisons across different information sources challenging.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Forward-looking limitations:</strong> Like all historical risk measures, the Ulcer Index makes no predictions about future drawdowns and may underestimate risk during regime changes.
                    </Typography>
                  </li>
                </ul>
              </Box>
            </Grid>
          </Grid>
        </Paper>

        {/* Comparison with Other Metrics */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>Comparison with Other Risk Metrics</Typography>
          <Grid container spacing={2}>
            <Grid item xs={12} md={6}>
              <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>
                <Typography variant="h6" gutterBottom>Ulcer Index vs. Standard Deviation</Typography>
                <Typography paragraph>
                  While <MuiLink component={Link} href="/docs/volatility">standard deviation</MuiLink> measures dispersion of returns around the mean (both positive and negative), the Ulcer Index only considers negative deviations from peak values. Standard deviation treats a 5% gain and a 5% loss as equivalent in risk terms, whereas the Ulcer Index completely ignores positive movements and focuses on the depth and duration of drawdowns.
                </Typography>
              </Box>
            </Grid>
            <Grid item xs={12} md={6}>
              <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>
                <Typography variant="h6" gutterBottom>Ulcer Index vs. Maximum Drawdown</Typography>
                <Typography paragraph>
                  <MuiLink component={Link} href="/docs/maximum-drawdown">Maximum drawdown</MuiLink> only captures the single worst peak-to-trough decline, while the Ulcer Index incorporates all drawdowns, their depths, and durations. Two portfolios might have identical maximum drawdowns of 20%, but if one recovered quickly while the other languished near the bottom for months, the Ulcer Index would be much higher for the second portfolio.
                </Typography>
              </Box>
            </Grid>
          </Grid>
        </Paper>

        {/* References */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>References</Typography>
          <ul>
            <li>
              <Typography paragraph><strong>Martin, P., & McCann, B. (1987)</strong>. "The Investor's Guide to Fidelity Funds: Winning Strategies for Mutual Fund Investors." <em>John Wiley & Sons</em>.</Typography>
            </li>
            <li>
              <Typography paragraph><strong>Martin, P. (2008)</strong>. "The Ulcer Index: Use it to avoid indigestion when investing." <em>Technical Analysis of Stocks & Commodities Magazine</em>, 26(7), 58.</Typography>
            </li>
            <li>
              <Typography paragraph><strong>Magdon-Ismail, M., & Atiya, A. (2004)</strong>. "Maximum drawdown." <em>Risk Magazine</em>, 17(10), 99-102.</Typography>
            </li>
            <li>
              <Typography paragraph><strong>Bacon, C. R. (2013)</strong>. <em>Practical Risk-Adjusted Performance Measurement</em>. Wiley.</Typography>
            </li>
            <li>
              <Typography paragraph><strong>Rollinger, T., & Hoffman, S. (2013)</strong>. "Sortino ratio: A better measure of risk." <em>Futures Magazine</em>, 1(2), 40-42.</Typography>
            </li>
          </ul>
        </Paper>

        {/* Related Topics */}
        <Paper elevation={2} sx={{ p: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>Related Topics</Typography>
          <Grid container spacing={2}>
            <Grid item xs={12} sm={6} md={4}>
              <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>
                <Typography variant="h6" gutterBottom>Calmar Ratio</Typography>
                <Typography variant="body2" paragraph>A performance measurement using the ratio of average annual compound rate of return to maximum drawdown.</Typography>
                <Link href="/docs/calmar-ratio" passHref>
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
                <Typography variant="h6" gutterBottom>Maximum Drawdown</Typography>
                <Typography variant="body2" paragraph>The largest peak-to-trough decline experienced in a portfolio's value over a specific time period.</Typography>
                <Link href="/docs/maximum-drawdown" passHref>
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

export default UlcerIndexPage; 