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
   COSKEWNESS PAGE                                                      
   ------------------------------------------------------------------- */
const CoskewnessPage: React.FC = () => {
  return (
    <>
      <Head>
        <title>Coskewness | Portfolio Optimisation</title>
        <meta
          name="description"
          content="A higher-moment risk measure that captures the relationship between an asset's returns and market volatility, enhancing portfolio construction beyond mean-variance analysis."
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
            Coskewness
          </Typography>
          <Typography variant="subtitle1" color="text.secondary">
            A third-moment measure that quantifies how an asset's returns interact with squared market returns, enhancing traditional mean-variance portfolio optimization.
          </Typography>
        </Box>

        {/* INTUITION */}
        <Paper elevation={3} sx={{ p: 3, my: 4 }}>
          <Typography variant="h5" gutterBottom>
            Why Coskewness?
          </Typography>
          <Typography paragraph>
            Traditional mean-variance optimization only considers the first two moments of return distributions—expected returns and 
            covariances. However, investors typically prefer positive skewness (larger probabilities of extreme positive returns) 
            and avoid negative skewness. <strong>Coskewness</strong> extends portfolio theory to include these preferences by 
            measuring how an asset's returns vary with squared market returns.
          </Typography>
          <Typography paragraph>
            Assets with positive coskewness tend to perform well when market volatility increases, providing a hedge against market 
            turbulence and commanding lower risk premiums. Conversely, assets with negative coskewness typically suffer during high 
            market volatility periods and require higher expected returns to compensate investors.
          </Typography>
        </Paper>

        {/* MATHEMATICS */}
        <Typography variant="h4" gutterBottom>
          1. Mathematical Definition
        </Typography>
        <Typography variant="h6" gutterBottom>
          1.1 Beyond Variance: The Third Moment
        </Typography>
        <Typography paragraph>
          Skewness measures the asymmetry of a probability distribution. For a return series <InlineMath math="r" />, skewness is defined as:
        </Typography>
        <Equation math="Skew(r) = \frac{\mathbb{E}[(r - \mu)^3]}{[\mathbb{E}[(r - \mu)^2]]^{3/2}} = \frac{\mathbb{E}[(r - \mu)^3]}{\sigma^3}" />
        <Typography paragraph>
          Where <InlineMath math="\mu" /> is the mean return and <InlineMath math="\sigma" /> is the standard deviation. Positive skewness 
          indicates an asymmetric tail extending toward more positive values, while negative skewness indicates an asymmetric tail 
          extending toward more negative values.
        </Typography>

        <Typography variant="h6" gutterBottom>
          1.2 Coskewness Definition
        </Typography>
        <Typography paragraph>
          Coskewness extends this concept to measure the relationship between an asset's return and squared market returns:
        </Typography>
        <Equation math="s_{i,m} = \frac{\mathbb{E}[(r_i - \mu_i)(r_m - \mu_m)^2]}{\sigma_i \sigma_m^2}" />
        <Typography paragraph>
          Where <InlineMath math="r_i" /> and <InlineMath math="r_m" /> are the returns of asset <InlineMath math="i" /> and the market, 
          <InlineMath math="\mu_i" /> and <InlineMath math="\mu_m" /> are their respective means, and <InlineMath math="\sigma_i" /> and 
          <InlineMath math="\sigma_m" /> are their standard deviations.
        </Typography>
        <Typography paragraph>
          In an unstandardized form, coskewness can be expressed as:
        </Typography>
        <Equation math="Coskew(r_i, r_m) = \mathbb{E}[(r_i - \mu_i)(r_m - \mu_m)^2]" />

        <Typography variant="h6" gutterBottom>
          1.3 Matrix Representation
        </Typography>
        <Typography paragraph>
          For a portfolio of <InlineMath math="n" /> assets, coskewness can be represented using a three-dimensional matrix:
        </Typography>
        <Equation math="S_{i,j,k} = \mathbb{E}[(r_i - \mu_i)(r_j - \mu_j)(r_k - \mu_k)]" />
        <Typography paragraph>
          The coskewness of a portfolio with weights <InlineMath math="w" /> can then be calculated as:
        </Typography>
        <Equation math="s_p = \sum_{i=1}^{n}\sum_{j=1}^{n}\sum_{k=1}^{n} w_i w_j w_k S_{i,j,k}" />
        <Typography paragraph>
          In practice, we often focus on the coskewness between each asset and the market portfolio, which simplifies the calculation.
        </Typography>

        <Typography variant="h6" gutterBottom>
          1.4 Estimation via Regression
        </Typography>
        <Typography paragraph>
          Coskewness can also be estimated using regression analysis:
        </Typography>
        <Equation math="r_i = \alpha + \beta r_m + \gamma r_m^2 + \epsilon" />
        <Typography paragraph>
          Where <InlineMath math="\gamma" /> captures the sensitivity of asset returns to squared market returns (coskewness), 
          <InlineMath math="\beta" /> is the traditional beta (systematic risk), and <InlineMath math="\alpha" /> is the intercept.
        </Typography>

        <Typography variant="h6" gutterBottom>
          1.5 Financial Significance
        </Typography>
        <Typography paragraph>
          Coskewness is financially significant for several reasons:
        </Typography>
        <ol>
          <li>
            <Typography paragraph>
              <strong>Pricing impact:</strong> Assets with negative coskewness (which tend to perform poorly when market volatility increases) 
              command higher risk premiums in equilibrium, as demonstrated in three-moment asset pricing models.
            </Typography>
          </li>
          <li>
            <Typography paragraph>
              <strong>Diversification benefits:</strong> Including assets with positive coskewness can improve a portfolio's risk profile 
              beyond what mean-variance optimization alone would achieve.
            </Typography>
          </li>
          <li>
            <Typography paragraph>
              <strong>Crisis hedging:</strong> Assets with positive coskewness can serve as partial hedges during market turbulence, 
              as they tend to perform relatively better when market volatility spikes.
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
                ["window", "252", "Rolling window length in trading days (~1 year)"],
                ["standardized", "True", "Whether to use standardized or raw coskewness"],
                ["method", "\"kraus-litzenberger\"", "Estimation method (kraus-litzenberger, harvey-siddique)"],
                ["min_periods", "60", "Minimum observations required for estimation"],
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
          When implementing coskewness in portfolio optimization:
        </Typography>
        <ul>
          <li>
            <Typography paragraph>
              <strong>Data requirements:</strong> Accurate coskewness estimation requires substantial historical data, as third-moment 
              statistics are more sensitive to sampling error than means and variances.
            </Typography>
          </li>
          <li>
            <Typography paragraph>
              <strong>Estimation method:</strong> The regression-based approach (Kraus-Litzenberger) is generally more stable and 
              interpretable than direct calculation, especially for small samples.
            </Typography>
          </li>
          <li>
            <Typography paragraph>
              <strong>Time-variation:</strong> Coskewness tends to vary over time, particularly during market regime changes. 
              Rolling-window or GARCH-based approaches can capture this time variation.
            </Typography>
          </li>
          <li>
            <Typography paragraph>
              <strong>Portfolio optimization:</strong> Including coskewness in portfolio optimization requires solving a cubic 
              programming problem, which is more complex than quadratic programming used in mean-variance optimization.
            </Typography>
          </li>
          <li>
            <Typography paragraph>
              <strong>Standardization:</strong> Standardized coskewness is comparable across assets and time periods, 
              while raw coskewness depends on the scale of returns.
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
                <li>Captures asymmetric risk not reflected in traditional mean-variance analysis.</li>
                <li>Provides insights into asset behavior during market volatility spikes.</li>
                <li>Helps identify potential hedges against market turbulence.</li>
                <li>Improves portfolio performance by including investor preferences for positive skewness.</li>
                <li>Addresses empirical anomalies unexplained by traditional CAPM.</li>
              </ul>
            </Box>
          </Grid>
          <Grid item xs={12} md={6}>
            <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>
              <Typography variant="h6" gutterBottom color="error">
                Limitations
              </Typography>
              <ul>
                <li>Requires substantial historical data for reliable estimation.</li>
                <li>More sensitive to outliers than mean and variance estimates.</li>
                <li>Complex to implement in portfolio optimization frameworks.</li>
                <li>Time-varying nature makes it challenging to use for long-term asset allocation.</li>
                <li>Limited consensus on the best estimation methodology.</li>
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
            <li>Harvey, C. R., & Siddique, A. (2000). <em>Conditional skewness in asset pricing tests</em>. <em>Journal of Finance</em>, 55(3), 1263-1295.</li>
            <li>Kraus, A., & Litzenberger, R. H. (1976). <em>Skewness preference and the valuation of risk assets</em>. <em>Journal of Finance</em>, 31(4), 1085-1100.</li>
            <li>Christoffersen, P., Feunou, B., Jacobs, K., & Turnbull, S. (2021). <em>Option-Based Estimation of the Price of Coskewness and Cokurtosis Risk</em>. <em>Journal of Financial and Quantitative Analysis</em>, 56(1), 65-91.</li>
            <li>Boudt, K., Cornilly, D., & Verdonck, T. (2020). <em>A coskewness shrinkage approach for estimating the skewness of linear combinations of random variables</em>. <em>Journal of Financial Econometrics</em>, 18(1), 1-23.</li>
            <li>Guidolin, M., & Timmermann, A. (2008). <em>International asset allocation under regime switching, skew, and kurtosis preferences</em>. <em>The Review of Financial Studies</em>, 21(2), 889-935.</li>
            <li>Martellini, L., & Ziemann, V. (2010). <em>Improved estimates of higher-order comoments and implications for portfolio selection</em>. <em>The Review of Financial Studies</em>, 23(4), 1467-1502.</li>
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
                  Cokurtosis
                </Typography>
                <Typography variant="body2" paragraph>
                  A fourth-moment measure that captures an asset's sensitivity to extreme market movements.
                </Typography>
                <Link href="/docs/cokurtosis" passHref>
                  <Button variant="contained" color="primary">
                    Learn More
                  </Button>
                </Link>
              </Box>
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>
                <Typography variant="h6" gutterBottom>
                  Skewness
                </Typography>
                <Typography variant="body2" paragraph>
                  A measure of the asymmetry of the probability distribution of returns about its mean.
                </Typography>
                <Link href="/docs/skewness" passHref>
                  <Button variant="contained" color="primary">
                    Learn More
                  </Button>
                </Link>
              </Box>
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>
                <Typography variant="h6" gutterBottom>
                  Kurtosis
                </Typography>
                <Typography variant="body2" paragraph>
                  A measure of the "tailedness" of the probability distribution indicating the presence of extreme values.
                </Typography>
                <Link href="/docs/kurtosis" passHref>
                  <Button variant="contained" color="primary">
                    Learn More
                  </Button>
                </Link>
              </Box>
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>
                <Typography variant="h6" gutterBottom>
                  Modern Portfolio Theory
                </Typography>
                <Typography variant="body2" paragraph>
                  Framework for constructing portfolios that maximize expected return for a given level of risk.
                </Typography>
                <Link href="/docs/modern-portfolio-theory" passHref>
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

export default CoskewnessPage; 