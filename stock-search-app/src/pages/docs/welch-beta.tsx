import React from "react";
import {
  Typography,
  Container,
  Box,
  Paper,
  Grid,
  Link as MuiLink,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Button,
  Card,
  CardContent,
  Divider
} from "@mui/material";
import Head from "next/head";
import Link from "next/link";
import "katex/dist/katex.min.css";
import { BlockMath, InlineMath } from "react-katex";
import TopNav from "../../components/TopNav";

/* -------------------------------------------------------------------
   EQUATION COMPONENT                                                    
   ------------------------------------------------------------------- */
const Equation: React.FC<{ math: string }> = ({ math }) => (
  <Box sx={{ p: 2, bgcolor: "#f5f5f5", borderRadius: 1, my: 2, textAlign: "center" }}>
    <BlockMath math={math} />
  </Box>
);

/* -------------------------------------------------------------------
   WELCH BETA PAGE                                                      
   ------------------------------------------------------------------- */
const WelchBetaPage: React.FC = () => {
  return (
    <>
      <Head>
        <title>Welch Beta (Slope-Winsorized Beta) | Portfolio Optimisation</title>
        <meta
          name="description"
          content="A robust market sensitivity measure using winsorized regression to dampen extreme return shocks, improving beta stability and out-of-sample performance."
        />
      </Head>

      <TopNav />

      <Container maxWidth="lg" sx={{ py: 6 }}>
        {/* NAVIGATION */}
        <Box sx={{ mb: 4, display: 'flex', gap: 2 }}>
          <Link href="/docs" passHref>
            <Button variant="outlined" color="primary">← Back to Docs</Button>
          </Link>
          <Link href="/" passHref>
            <Button variant="outlined" color="secondary">← Back to Portfolio Optimizer</Button>
          </Link>
        </Box>

        {/* HEADER */}
        <Box sx={{ mb: 6, textAlign: 'center' }}>
          <Typography variant="h3" component="h1" gutterBottom>
            Welch Beta (Slope‑Winsorized Beta)
          </Typography>
          <Typography variant="subtitle1" color="text.secondary">
            A noise‑resistant alternative to OLS beta estimates by winsorizing extreme return observations before regression.
          </Typography>
        </Box>

        {/* INTUITION */}
        <Paper elevation={3} sx={{ p: 3, my: 4 }}>
          <Typography variant="h5" gutterBottom>
            Why Welch Beta?
          </Typography>
          <Typography paragraph>
            Traditional beta from OLS regression can be unduly influenced by market crashes,
            outlier returns, or short‑sample noise, leading to unstable estimates.
            <strong> Welch Beta</strong> applies winsorization to both asset and market return
            series at chosen quantiles, trimming extremes and yielding a slope that
            better reflects typical co‑movement between assets and the market.
          </Typography>
          <Typography paragraph>
            By reducing the impact of extreme observations, Welch Beta provides more reliable 
            risk estimates that are less likely to change dramatically during volatile market periods, 
            resulting in more stable portfolio construction and improved out-of-sample performance.
          </Typography>
        </Paper>

        {/* MATHEMATICS */}
        <Typography variant="h4" gutterBottom>
          1. Mathematical Definition
        </Typography>
        <Typography variant="h6" gutterBottom>
          1.1 The Problem with Standard Beta
        </Typography>
        <Typography paragraph>
          The traditional beta coefficient is defined as:
        </Typography>
        <Equation math="\beta = \frac{\operatorname{Cov}(r_i,\,r_m)}{\operatorname{Var}(r_m)}" />
        <Typography paragraph>
          Where <InlineMath math="r_i" /> represents the excess returns of asset <InlineMath math="i" /> and <InlineMath math="r_m" /> represents 
          the excess returns of the market. When estimated via Ordinary Least Squares (OLS), beta minimizes the squared residuals:
        </Typography>
        <Equation math="\min_{\alpha, \beta} \sum_{t=1}^{T} (r_{i,t} - \alpha - \beta r_{m,t})^2" />
        <Typography paragraph>
          The problem arises because squared residuals heavily penalize outliers, making the beta estimate disproportionately 
          influenced by extreme market movements that may not reflect normal asset-market relationships.
        </Typography>

        <Typography variant="h6" gutterBottom>
          1.2 Winsorization Process
        </Typography>
        <Typography paragraph>
          Given return series <InlineMath math="r_i" /> for the asset and <InlineMath math="r_m" />
          for the market, define lower and upper cutoff quantiles <InlineMath math="\alpha" /> and
          <InlineMath math="1-\alpha" />. Winsorized returns <InlineMath math="\tilde r" /> are:
        </Typography>
        <Equation math="\tilde r_t = \begin{cases}F^{-1}(\alpha), & r_t < F^{-1}(\alpha)\\r_t, & F^{-1}(\alpha)\le r_t \le F^{-1}(1-\alpha)\\F^{-1}(1-\alpha), & r_t > F^{-1}(1-\alpha)\end{cases}" />
        <Typography paragraph>
          where <InlineMath math="F^{-1}(q)" /> is the empirical quantile at probability <InlineMath math="q" />. This transformation 
          replaces values below the <InlineMath math="\alpha" /> percentile with the <InlineMath math="\alpha" /> percentile value, 
          and values above the <InlineMath math="1-\alpha" /> percentile with the <InlineMath math="1-\alpha" /> percentile value.
        </Typography>
        <Typography paragraph>
          For example, with <InlineMath math="\alpha = 0.01" />, the bottom 1% of returns are replaced with the 1st percentile value, 
          and the top 1% are replaced with the 99th percentile value, effectively dampening the impact of extreme observations.
        </Typography>

        <Typography variant="h6" gutterBottom>
          1.3 Winsorized Regression
        </Typography>
        <Typography paragraph>
          After winsorizing both the asset and market returns, the Welch Beta is estimated via OLS on the winsorized data:
        </Typography>
        <Equation math="\beta_{w} = \frac{\operatorname{Cov}(\tilde r_i,\,\tilde r_m)}{\operatorname{Var}(\tilde r_m)}" />
        <Typography paragraph>
          This can also be expressed in regression form:
        </Typography>
        <Equation math="\tilde r_{i,t} = \alpha + \beta_w \tilde r_{m,t} + \epsilon_t" />
        <Typography paragraph>
          Where <InlineMath math="\tilde r_{i,t}" /> and <InlineMath math="\tilde r_{m,t}" /> are the winsorized returns for the asset and market at time <InlineMath math="t" />, 
          and <InlineMath math="\epsilon_t" /> is the error term.
        </Typography>

        <Typography variant="h6" gutterBottom>
          1.4 Financial Significance
        </Typography>
        <Typography paragraph>
          The Welch Beta provides three key financial benefits:
        </Typography>
        <ol>
          <li>
            <Typography paragraph>
              <strong>Stability:</strong> By limiting the influence of extreme returns, the beta estimate becomes more stable over time, 
              resulting in lower portfolio turnover and transaction costs.
            </Typography>
          </li>
          <li>
            <Typography paragraph>
              <strong>Representativeness:</strong> The winsorized beta better captures the normal relationship between asset and market returns, 
              improving the accuracy of risk measurement under typical market conditions.
            </Typography>
          </li>
          <li>
            <Typography paragraph>
              <strong>Forward-looking accuracy:</strong> Research by Welch (2022) shows that winsorized betas have better out-of-sample 
              predictive power for future beta values compared to traditional OLS betas.
            </Typography>
          </li>
        </ol>

        {/* DEFAULT PARAMS */}
        <Typography variant="h4" gutterBottom>
          2. Default Parameters
        </Typography>
        <TableContainer component={Paper} sx={{ mb: 4 }}>
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell>Parameter</TableCell>
                <TableCell>Default</TableCell>
                <TableCell>Description</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {[
                ["alpha", "0.01", "Winsorization tail probability (typically 0.01 or 0.05)"],
                ["window", "252", "Rolling window length in trading days (~ 1 year)"],
                ["method", "\"winsorized\"", "Type of trimming (winsorized vs. truncated)"],
                ["rf", "0.0", "Risk-free rate subtracted from returns before estimation"],
              ].map(([param, def, desc]) => (
                <TableRow key={param as string}>
                  <TableCell>{param}</TableCell>
                  <TableCell>
                    <code>{def as string}</code>
                  </TableCell>
                  <TableCell>{desc}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>

        {/* IMPLEMENTATION CONSIDERATIONS */}
        <Typography variant="h4" gutterBottom>
          3. Implementation Considerations
        </Typography>
        <Typography paragraph>
          When implementing Welch Beta in portfolio optimization:
        </Typography>
        <ul>
          <li>
            <Typography paragraph>
              <strong>Choice of α:</strong> Typical values range from 0.01 (1%) to 0.05 (5%). Lower values preserve more data 
              but offer less protection against outliers. Higher values provide more smoothing but may discard valuable information.
            </Typography>
          </li>
          <li>
            <Typography paragraph>
              <strong>Rolling windows:</strong> Welch Beta can be calculated over rolling windows to capture time-varying sensitivity. 
              A common window length is 252 trading days (approximately one year).
            </Typography>
          </li>
          <li>
            <Typography paragraph>
              <strong>Winsorization vs. Truncation:</strong> Winsorization replaces extreme values with percentile bounds, 
              while truncation removes them entirely. Winsorization is generally preferred as it preserves sample size.
            </Typography>
          </li>
          <li>
            <Typography paragraph>
              <strong>Market definition:</strong> The choice of market index (e.g., Nifty 50 vs. Sensex) can affect beta estimates. 
              Choose an index that best represents the investment universe.
            </Typography>
          </li>
        </ul>

        {/* PROS / CONS */}
        <Typography variant="h4" gutterBottom>
          4. Advantages and Limitations
        </Typography>
        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>
              <Typography variant="h6" gutterBottom color="primary">
                Advantages
              </Typography>
              <ul>
                <li>Reduces sensitivity to market crashes and return spikes.</li>
                <li>Provides more stable rolling estimates with lower turnover.</li>
                <li>Improves out-of-sample beta prediction accuracy.</li>
                <li>Easy to implement with standard statistical packages.</li>
                <li>Computationally efficient compared to GARCH or other time-varying methods.</li>
              </ul>
            </Box>
          </Grid>
          <Grid item xs={12} md={6}>
            <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>
              <Typography variant="h6" gutterBottom color="error">
                Limitations
              </Typography>
              <ul>
                <li>Choice of winsorization threshold (<InlineMath math="\alpha" />) is somewhat subjective.</li>
                <li>Ignores potentially valuable information in extreme return tails.</li>
                <li>May understate true systematic risk during turbulent market regimes.</li>
                <li>Does not explicitly model time-varying volatility like GARCH models.</li>
                <li>Assumes symmetrical treatment of positive and negative return outliers.</li>
              </ul>
            </Box>
          </Grid>
        </Grid>

        {/* REFERENCES */}
        <Paper elevation={2} sx={{ p: 4, mb: 4, mt: 4 }}>
          <Typography variant="h4" gutterBottom>
            5. References
          </Typography>
          <ul>
            <li>Welch, I. (2022). <em>Simply Better Market Betas</em>. <em>Critical Finance Review</em>, 11(2), 207–244.</li>
            <li>Welch, I. (2019). <em>Simpler Better Market Betas</em>. NBER Working Paper No. 26105.</li>
            <li>Levi, Y., & Welch, I. (2020). <em>Symmetric and Asymmetric Market Betas and Downside Risk</em>. <em>Review of Financial Studies</em>, 33(6), 2772–2795.</li>
            <li>Knif, J., Kolari, J., & Pynnönen, S. (2013). <em>The Impact of Outliers on the Time-Stability of Beta in the Finnish Stock Market</em>. <em>Journal of Applied Statistics</em>, 40(5), 968-980.</li>
            <li>Hampel, F. R., Ronchetti, E. M., Rousseeuw, P. J., & Stahel, W. A. (2011). <em>Robust Statistics: The Approach Based on Influence Functions</em>. John Wiley & Sons.</li>
          </ul>
        </Paper>

        {/* RELATED TOPICS */}
        <Paper elevation={2} sx={{ p: 4, mb: 4 }}>
          <Typography variant="h4" gutterBottom>
            Related Topics
          </Typography>
          <Grid container spacing={2}>
            <Grid item xs={12} sm={6} md={3}>
              <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>
                <Typography variant="h6" gutterBottom>
                  Semi Beta
                </Typography>
                <Typography variant="body2" paragraph>
                  A downside beta that measures portfolio sensitivity to the benchmark only during down markets.
                </Typography>
                <Link href="/docs/semi-beta" passHref>
                  <Button variant="contained" color="primary">
                    Learn More
                  </Button>
                </Link>
              </Box>
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>
                <Typography variant="h6" gutterBottom>
                  GARCH Beta
                </Typography>
                <Typography variant="body2" paragraph>
                  A time-varying measure of portfolio beta that accounts for volatility clustering using GARCH models.
                </Typography>
                <Link href="/docs/garch-beta" passHref>
                  <Button variant="contained" color="primary">
                    Learn More
                  </Button>
                </Link>
              </Box>
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>
                <Typography variant="h6" gutterBottom>
                  Blume-Adjusted Beta
                </Typography>
                <Typography variant="body2" paragraph>
                  A modified beta calculation that adjusts for the tendency of betas to revert toward the market average over time.
                </Typography>
                <Link href="/docs/blume-adjusted-beta" passHref>
                  <Button variant="contained" color="primary">
                    Learn More
                  </Button>
                </Link>
              </Box>
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>
                <Typography variant="h6" gutterBottom>
                  Portfolio Beta
                </Typography>
                <Typography variant="body2" paragraph>
                  Traditional measure of systematic risk that represents how an asset moves relative to the overall market.
                </Typography>
                <Link href="/docs/capm-beta" passHref>
                  <Button variant="contained" color="primary">
                    Learn More
                  </Button>
                </Link>
              </Box>
            </Grid>
          </Grid>
        </Paper>
      </Container>
    </>
  );
};

export default WelchBetaPage; 