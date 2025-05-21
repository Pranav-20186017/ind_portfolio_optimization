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

const MaximumDrawdownPage: React.FC = () => {
  return (
    <>
      <Head>
        <title>Maximum Drawdown | Portfolio Optimization</title>
        <meta
          name="description"
          content="Learn about Maximum Drawdown, a measure of the largest peak-to-trough decline in a portfolio's value, representing the worst-case scenario for an investment."
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
            Maximum Drawdown
          </Typography>
          <Typography variant="subtitle1" color="text.secondary">
            Measuring the worst peak-to-trough decline in portfolio value
          </Typography>
        </Box>

        {/* Overview */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>Overview</Typography>
          <Typography paragraph>
            <strong>Maximum Drawdown (MDD)</strong> is a key risk measure that captures the largest percentage drop in a portfolio's value from a peak to the subsequent trough before a new peak is established. It represents the worst-case historical loss scenario an investor would have experienced had they invested at the most unfortunate time and sold at the worst possible moment.
          </Typography>
          <Typography paragraph>
            As one of the most intuitive and widely used measures of downside risk, maximum drawdown is particularly valuable for risk-averse investors, as it quantifies the steepest decline they might have to endure. Unlike volatility measures that treat upside and downside movements equally, maximum drawdown focuses exclusively on the largest sustained loss, providing a clear picture of downside risk.
          </Typography>
        </Paper>

        {/* Intuitive Explanation */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>Intuitive Explanation</Typography>
          <Typography paragraph>
            Think of maximum drawdown as the "biggest drop" your portfolio could experience based on historical data. It answers the question: "What's the worst-case scenario I might face if I invest today?"
          </Typography>
          <Typography paragraph>
            Imagine you're hiking on a mountain trail that has both uphill and downhill segments. Maximum drawdown is like measuring the steepest and longest downhill section of the entire trail—the part where you descend the most from a high point before starting to climb up again. For investors, this represents the period where they would lose the most money before the portfolio begins to recover.
          </Typography>
          <Box sx={{ bgcolor: 'background.paper', p: 2, borderRadius: 1, border: '1px solid #e0e0e0', mt: 2 }}>
            <Typography variant="body1">
              <strong>Mountain climbing analogy:</strong> If your investment journey is like climbing a mountain with multiple peaks and valleys, maximum drawdown measures the deepest valley you would have to traverse—from the highest peak you've reached to the lowest point before you start climbing again. The larger this valley, the more nerve-wracking the journey becomes, and the more likely investors might abandon their climb altogether.
            </Typography>
          </Box>
        </Paper>

        {/* Mathematical Explanation */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>Detailed Mathematical Explanation</Typography>

          <Typography paragraph>
            Maximum Drawdown is calculated by finding the largest percentage decline between a peak and the subsequent trough in the value of a portfolio over a specific time period:
          </Typography>

          <Box sx={{ p: 3, bgcolor: 'background.paper', borderRadius: 1, border: '1px solid #e0e0e0', my: 3 }}>
            <Typography variant="subtitle1" gutterBottom><strong>Maximum Drawdown Formula</strong></Typography>
            <Equation math="\text{MDD} = \min_{t \in (0,T)} \left( \frac{V_t - \max_{s \in (0,t)} V_s}{\max_{s \in (0,t)} V_s} \right)" />
            <Typography variant="body2">
              where <InlineMath math="V_t" /> is the value of the portfolio at time <InlineMath math="t" />, and <InlineMath math="T" /> is the total time period being analyzed.
            </Typography>
          </Box>

          <Typography paragraph>
            In simpler terms, for each point in time, we:
          </Typography>
          <ol>
            <li>
              <Typography paragraph>
                Find the maximum value the portfolio has reached up to that point (the peak)
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                Calculate the percentage decline from that peak to the current value
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                Identify the largest percentage decline across all time periods
              </Typography>
            </li>
          </ol>

          <Typography variant="h6" gutterBottom sx={{ mt: 3 }}>Alternative Formulation</Typography>
          <Typography paragraph>
            Maximum drawdown can also be expressed using drawdown (DD) values:
          </Typography>

          <Equation math="\text{DD}_t = \frac{V_t - \max_{s \in (0,t)} V_s}{\max_{s \in (0,t)} V_s}" />
          <Typography paragraph>
            Then, the maximum drawdown is simply:
          </Typography>
          <Equation math="\text{MDD} = \min_{t \in (0,T)} \text{DD}_t" />

          <Typography paragraph>
            Since drawdown values are always zero or negative (as the current value is always less than or equal to the peak value), the maximum drawdown is reported as a positive percentage to indicate the magnitude of the decline.
          </Typography>

          <Typography variant="h6" gutterBottom sx={{ mt: 3 }}>Calmar Ratio</Typography>
          <Typography paragraph>
            Maximum drawdown is an important component of the <MuiLink component={Link} href="/education/calmar-ratio">Calmar Ratio</MuiLink>, which measures the relationship between returns and risk:
          </Typography>

          <Equation math="\text{Calmar Ratio} = \frac{\text{Annualized Return}}{\text{Maximum Drawdown}}" />

          <Typography paragraph>
            This ratio helps investors understand how much return they're getting for taking on the risk of a potential significant drawdown.
          </Typography>
        </Paper>

        {/* Implementation Details */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>Implementation in Our Service</Typography>
          <Typography paragraph>
            Our portfolio analyzer calculates Maximum Drawdown through the following steps:
          </Typography>
          <ul>
            <li>
              <Typography paragraph>
                <strong>Historical Price Series:</strong> We track the value of a portfolio or asset over the specified time period.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Running Maximum:</strong> For each point in time, we identify the highest portfolio value achieved up to that point.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Drawdown Calculation:</strong> We calculate the percentage decline from the peak to each subsequent point.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Maximum Identification:</strong> We identify the largest drawdown over the entire period.
              </Typography>
            </li>
          </ul>
          <Typography paragraph>
            We display Maximum Drawdown as a percentage, often accompanied by a visual representation showing when the drawdown occurred and how long it took for the portfolio to recover. This visualization helps investors understand not just the magnitude of potential losses but also their duration.
          </Typography>
          <Box sx={{ p: 2, bgcolor: 'background.paper', borderRadius: 1, border: '1px solid #e0e0e0', mt: 2, textAlign: 'center' }}>
            <Typography variant="subtitle2" gutterBottom>Maximum Drawdown Visualization (Placeholder)</Typography>
            <Box sx={{ height: '300px', bgcolor: '#f5f5f5', display: 'flex', alignItems: 'center', justifyContent: 'center', mb: 2 }}>
              <Typography variant="body2" color="text.secondary">
                [Placeholder for Maximum Drawdown visualization chart]
              </Typography>
            </Box>
            <Typography variant="caption" color="text.secondary">
              The chart shows the portfolio value over time, highlighting the maximum drawdown period from peak to trough, and the recovery time.
            </Typography>
          </Box>
        </Paper>

        {/* Worked Example */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>Worked Example</Typography>
          <Typography paragraph>
            Let's calculate the Maximum Drawdown for a portfolio with the following monthly values over a 12-month period:
          </Typography>
          <Typography paragraph>
            $100, $105, $110, $108, $112, $105, $98, $95, $100, $104, $108, $112
          </Typography>

          <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>Step 1: Identify the running maximum at each point</Typography>
          <Typography paragraph>
            $100, $105, $110, $110, $112, $112, $112, $112, $112, $112, $112, $112
          </Typography>

          <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>Step 2: Calculate drawdowns at each point</Typography>
          <ul>
            <li><Typography paragraph>Month 1: ($100 - $100)/$100 = 0% (no drawdown)</Typography></li>
            <li><Typography paragraph>Month 2: ($105 - $105)/$105 = 0% (no drawdown)</Typography></li>
            <li><Typography paragraph>Month 3: ($110 - $110)/$110 = 0% (no drawdown)</Typography></li>
            <li><Typography paragraph>Month 4: ($108 - $110)/$110 = -1.82%</Typography></li>
            <li><Typography paragraph>Month 5: ($112 - $112)/$112 = 0% (new peak)</Typography></li>
            <li><Typography paragraph>Month 6: ($105 - $112)/$112 = -6.25%</Typography></li>
            <li><Typography paragraph>Month 7: ($98 - $112)/$112 = -12.50%</Typography></li>
            <li><Typography paragraph>Month 8: ($95 - $112)/$112 = -15.18% (largest drawdown)</Typography></li>
            <li><Typography paragraph>Month 9: ($100 - $112)/$112 = -10.71%</Typography></li>
            <li><Typography paragraph>Month 10: ($104 - $112)/$112 = -7.14%</Typography></li>
            <li><Typography paragraph>Month 11: ($108 - $112)/$112 = -3.57%</Typography></li>
            <li><Typography paragraph>Month 12: ($112 - $112)/$112 = 0% (full recovery to previous peak)</Typography></li>
          </ul>

          <Typography variant="h6" gutterBottom sx={{ mt: 2 }}>Step 3: Identify the Maximum Drawdown</Typography>
          <Typography paragraph>
            The largest drawdown occurred in month 8, with a value of -15.18%.
          </Typography>
          <Typography paragraph>
            Therefore, the Maximum Drawdown (MDD) = 15.18%
          </Typography>

          <Typography paragraph>
            This means that an investor who bought at the peak ($112 in month 5) and sold at the trough ($95 in month 8) would have lost 15.18% of their investment. It took 4 months for the portfolio to recover from this drawdown and return to its previous peak.
          </Typography>
        </Paper>

        {/* Applications */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>Practical Applications</Typography>
          <Typography paragraph>
            Maximum Drawdown serves several important purposes in portfolio management and risk assessment:
          </Typography>
          <ul>
            <li>
              <Typography paragraph>
                <strong>Risk Assessment:</strong> Maximum Drawdown helps investors understand the worst-case historical scenario, allowing them to gauge whether they could emotionally and financially withstand such a decline.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Strategy Comparison:</strong> When comparing investment strategies with similar returns, Maximum Drawdown helps identify which strategies would have been less painful during market downturns.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Portfolio Construction:</strong> By minimizing Maximum Drawdown as an objective in portfolio optimization, investors can build portfolios designed to weather severe market conditions.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Risk Management:</strong> Setting stop-loss orders or rebalancing triggers based on drawdown thresholds can help manage risk and preserve capital during market declines.
              </Typography>
            </li>
            <li>
              <Typography paragraph>
                <strong>Retirement Planning:</strong> For retirees making regular withdrawals, large drawdowns early in retirement can significantly impact portfolio longevity (sequence of returns risk). Understanding Maximum Drawdown helps plan for this risk.
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
                      <strong>Intuitive interpretation:</strong> Maximum Drawdown is easy to understand as "the biggest loss you could have experienced."
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Captures tail risk:</strong> Focuses on extreme negative events rather than average behavior, aligning with investor psychology.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Non-parametric:</strong> Makes no assumptions about the distribution of returns, making it suitable for asymmetric or non-normal return patterns.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Temporal awareness:</strong> Unlike point-in-time measures, Maximum Drawdown captures a sequence of returns and the persistence of losses.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Behavioral relevance:</strong> Directly addresses the risk that investors will panic and sell during severe downturns.
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
                      <strong>Sample dependency:</strong> Only captures historical drawdowns, potentially missing future drawdown scenarios not present in the sample period.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Singular focus:</strong> Represents only the worst case, ignoring other significant but smaller drawdowns that might also impact investor behavior.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>No recovery information:</strong> The Maximum Drawdown value alone doesn't indicate how long the recovery took, which is also important for investors.
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Time period sensitivity:</strong> Results can vary significantly based on the chosen analysis period and can be dominated by a single extreme event (like the 2008 crisis).
                    </Typography>
                  </li>
                  <li>
                    <Typography paragraph>
                      <strong>Doesn't capture frequency:</strong> A portfolio with one severe drawdown might have the same Maximum Drawdown as one with multiple moderate drawdowns.
                    </Typography>
                  </li>
                </ul>
              </Box>
            </Grid>
          </Grid>
        </Paper>

        {/* Comparison with Other Risk Metrics */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h5" component="h2" gutterBottom>Comparison with Other Risk Metrics</Typography>
          <Grid container spacing={2}>
            <Grid item xs={12} md={6}>
              <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>
                <Typography variant="h6" gutterBottom>Maximum Drawdown vs. Volatility</Typography>
                <Typography paragraph>
                  While <MuiLink component={Link} href="/education/volatility">volatility</MuiLink> measures the dispersion of returns around their mean (treating upside and downside movements equally), Maximum Drawdown focuses exclusively on the worst sustained loss. A portfolio might have low volatility but still experience a significant drawdown during a specific period. Conversely, a portfolio with high volatility might have smaller maximum drawdowns if its swings are more evenly distributed between gains and losses.
                </Typography>
              </Box>
            </Grid>
            <Grid item xs={12} md={6}>
              <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>
                <Typography variant="h6" gutterBottom>Maximum Drawdown vs. Ulcer Index</Typography>
                <Typography paragraph>
                  The <MuiLink component={Link} href="/education/ulcer-index">Ulcer Index</MuiLink> incorporates all drawdowns (not just the maximum) and considers both their depth and duration. It provides a more comprehensive view of downside risk by penalizing portfolios that remain underwater for extended periods. Maximum Drawdown captures only the single worst event, while the Ulcer Index tracks the overall "pain" experienced across all drawdown periods.
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
              <Typography paragraph><strong>Magdon-Ismail, M., & Atiya, A. (2004)</strong>. "Maximum drawdown." <em>Risk Magazine</em>, 17(10), 99-102.</Typography>
            </li>
            <li>
              <Typography paragraph><strong>Chekhlov, A., Uryasev, S., & Zabarankin, M. (2005)</strong>. "Drawdown measure in portfolio optimization." <em>International Journal of Theoretical and Applied Finance</em>, 8(01), 13-58.</Typography>
            </li>
            <li>
              <Typography paragraph><strong>Bacon, C. R. (2013)</strong>. <em>Practical Risk-Adjusted Performance Measurement</em>. Wiley.</Typography>
            </li>
            <li>
              <Typography paragraph><strong>Burghardt, G., & Liu, L. (2013)</strong>. "It's the autocorrelation, stupid." <em>The Journal of Derivatives</em>, 21(1), 6-16.</Typography>
            </li>
            <li>
              <Typography paragraph><strong>Grossman, S. J., & Zhou, Z. (1993)</strong>. "Optimal investment strategies for controlling drawdowns." <em>Mathematical Finance</em>, 3(3), 241-276.</Typography>
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
                <Link href="/education/calmar-ratio" passHref>
                  <Button variant="contained" color="primary">Learn More</Button>
                </Link>
              </Box>
            </Grid>
            <Grid item xs={12} sm={6} md={4}>
              <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>
                <Typography variant="h6" gutterBottom>Ulcer Index</Typography>
                <Typography variant="body2" paragraph>A volatility measure that captures the depth and duration of drawdowns, focusing on downside movement.</Typography>
                <Link href="/education/ulcer-index" passHref>
                  <Button variant="contained" color="primary">Learn More</Button>
                </Link>
              </Box>
            </Grid>
            <Grid item xs={12} sm={6} md={4}>
              <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>
                <Typography variant="h6" gutterBottom>Volatility</Typography>
                <Typography variant="body2" paragraph>A statistical measure of the dispersion of returns, usually measured using standard deviation.</Typography>
                <Link href="/education/volatility" passHref>
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

export default MaximumDrawdownPage; 