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
   COKURTOSIS PAGE                                                      
   ------------------------------------------------------------------- */
const CokurtosisPage: React.FC = () => {
  return (
    <>
      <Head>
        <title>Cokurtosis | Portfolio Optimisation</title>
        <meta
          name="description"
          content="A fourth-moment risk measure that captures an asset's sensitivity to extreme market movements, enhancing tail risk management in portfolio construction."
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
            Cokurtosis
          </Typography>
          <Typography variant="subtitle1" color="text.secondary">
            A fourth-moment measure that quantifies how an asset's returns interact with extreme market movements, extending portfolio optimization beyond mean-variance-skewness frameworks.
          </Typography>
        </Box>

        {/* INTUITION */}
        <Paper elevation={3} sx={{ p: 3, my: 4 }}>
          <Typography variant="h5" gutterBottom>
            Why Cokurtosis?
          </Typography>
          <Typography paragraph>
            Traditional portfolio theory focuses on the first two moments of return distributions, while more advanced approaches 
            incorporate the third moment (skewness). <strong>Cokurtosis</strong> extends this further by capturing the fourth 
            moment, which measures the propensity for extreme outcomes—both positive and negative—and the "fatness" of the return 
            distribution tails.
          </Typography>
          <Typography paragraph>
            Assets with high positive cokurtosis tend to amplify extreme market movements, potentially exacerbating portfolio losses 
            during market crashes. Conversely, assets with low or negative cokurtosis can provide a cushioning effect during extreme 
            market events, making them valuable for tail risk management and crisis-resilient portfolio construction.
          </Typography>
        </Paper>

        {/* MATHEMATICS */}
        <Typography variant="h4" gutterBottom>
          1. Mathematical Definition
        </Typography>
        <Typography variant="h6" gutterBottom>
          1.1 Kurtosis: The Fourth Moment
        </Typography>
        <Typography paragraph>
          Kurtosis measures the "tailedness" of a probability distribution. For a return series <InlineMath math="r" />, kurtosis is defined as:
        </Typography>
        <Equation math="Kurt(r) = \frac{\mathbb{E}[(r - \mu)^4]}{[\mathbb{E}[(r - \mu)^2]]^{2}} = \frac{\mathbb{E}[(r - \mu)^4]}{\sigma^4}" />
        <Typography paragraph>
          Where <InlineMath math="\mu" /> is the mean return and <InlineMath math="\sigma" /> is the standard deviation. A distribution 
          with kurtosis greater than 3 (the kurtosis of a normal distribution) is called leptokurtic and has fatter tails, indicating 
          a higher probability of extreme outcomes.
        </Typography>

        <Typography variant="h6" gutterBottom>
          1.2 Cokurtosis Definition
        </Typography>
        <Typography paragraph>
          Cokurtosis extends this concept to measure the co-movement between an asset's returns and cubed market returns:
        </Typography>
        <Equation math="k_{i,m} = \frac{\mathbb{E}[(r_i - \mu_i)(r_m - \mu_m)^3]}{\sigma_i \sigma_m^3}" />
        <Typography paragraph>
          Where <InlineMath math="r_i" /> and <InlineMath math="r_m" /> are the returns of asset <InlineMath math="i" /> and the market, 
          <InlineMath math="\mu_i" /> and <InlineMath math="\mu_m" /> are their respective means, and <InlineMath math="\sigma_i" /> and 
          <InlineMath math="\sigma_m" /> are their standard deviations.
        </Typography>
        <Typography paragraph>
          In an unstandardized form, cokurtosis can be expressed as:
        </Typography>
        <Equation math="Cokurt(r_i, r_m) = \mathbb{E}[(r_i - \mu_i)(r_m - \mu_m)^3]" />

        <Typography variant="h6" gutterBottom>
          1.3 Matrix Representation
        </Typography>
        <Typography paragraph>
          For a portfolio of <InlineMath math="n" /> assets, cokurtosis can be represented using a four-dimensional matrix:
        </Typography>
        <Equation math="K_{i,j,k,l} = \mathbb{E}[(r_i - \mu_i)(r_j - \mu_j)(r_k - \mu_k)(r_l - \mu_l)]" />
        <Typography paragraph>
          The cokurtosis of a portfolio with weights <InlineMath math="w" /> can then be calculated as:
        </Typography>
        <Equation math="k_p = \sum_{i=1}^{n}\sum_{j=1}^{n}\sum_{k=1}^{n}\sum_{l=1}^{n} w_i w_j w_k w_l K_{i,j,k,l}" />
        <Typography paragraph>
          In practice, we often focus on the cokurtosis between each asset and the market portfolio for computational tractability.
        </Typography>

        <Typography variant="h6" gutterBottom>
          1.4 Estimation via Regression
        </Typography>
        <Typography paragraph>
          Similar to coskewness, cokurtosis can be estimated using regression analysis:
        </Typography>
        <Equation math="r_i = \alpha + \beta r_m + \gamma r_m^2 + \delta r_m^3 + \epsilon" />
        <Typography paragraph>
          Where <InlineMath math="\delta" /> captures the asset's sensitivity to cubed market returns (cokurtosis), alongside 
          the traditional beta <InlineMath math="\beta" /> and coskewness <InlineMath math="\gamma" /> coefficients.
        </Typography>

        <Typography variant="h6" gutterBottom>
          1.5 Financial Significance
        </Typography>
        <Typography paragraph>
          Cokurtosis is financially significant for several reasons:
        </Typography>
        <ol>
          <li>
            <Typography paragraph>
              <strong>Tail risk management:</strong> Cokurtosis directly measures an asset's contribution to portfolio tail risk, 
              helping identify securities that might amplify losses during market crashes.
            </Typography>
          </li>
          <li>
            <Typography paragraph>
              <strong>Pricing impact:</strong> Assets with high positive cokurtosis (which tend to exacerbate extreme market movements) 
              may command higher risk premiums in a four-moment asset pricing framework.
            </Typography>
          </li>
          <li>
            <Typography paragraph>
              <strong>Crisis resilience:</strong> Building portfolios with controlled cokurtosis can enhance resilience to market crashes 
              and extreme events beyond what traditional diversification achieves.
            </Typography>
          </li>
          <li>
            <Typography paragraph>
              <strong>Options pricing:</strong> Cokurtosis helps explain the volatility smile observed in options markets, as it captures 
              the non-normality of return distributions that affects option values.
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
                ["standardized", "True", "Whether to use standardized or raw cokurtosis"],
                ["method", "\"dittmar\"", "Estimation method (dittmar, fang-lai)"],
                ["min_periods", "100", "Minimum observations required for estimation"],
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
          When implementing cokurtosis in portfolio optimization:
        </Typography>
        <ul>
          <li>
            <Typography paragraph>
              <strong>Data requirements:</strong> Fourth-moment statistics require substantial historical data for reliable estimation—typically 
              two to three years of daily returns at minimum. Longer periods provide more stable estimates but may include outdated market regimes.
            </Typography>
          </li>
          <li>
            <Typography paragraph>
              <strong>Estimation precision:</strong> Cokurtosis is even more sensitive to outliers and estimation error than coskewness. 
              Shrinkage estimators or robust statistics can improve reliability.
            </Typography>
          </li>
          <li>
            <Typography paragraph>
              <strong>Computational complexity:</strong> The full cokurtosis tensor for a portfolio of <InlineMath math="n" /> assets has 
              <InlineMath math="n^4" /> elements, making it computationally intensive for large portfolios. Market-based simplifications 
              reduce this to <InlineMath math="n" /> calculations.
            </Typography>
          </li>
          <li>
            <Typography paragraph>
              <strong>Optimization challenges:</strong> Including cokurtosis in portfolio optimization leads to quartic programming problems, 
              which are more complex than quadratic (mean-variance) or cubic (mean-variance-skewness) problems.
            </Typography>
          </li>
          <li>
            <Typography paragraph>
              <strong>Empirical relevance:</strong> While theoretically important, the empirical significance of cokurtosis varies across 
              markets and time periods. Test its relevance in your specific investment universe before implementation.
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
                <li>Captures tail risk beyond what variance and skewness measures.</li>
                <li>Helps identify assets that amplify or dampen extreme market movements.</li>
                <li>Enhances crisis-period portfolio management and tail hedging.</li>
                <li>Aligns with investor preferences for avoiding extreme negative outcomes.</li>
                <li>Improves model fit for non-normal return distributions common in financial markets.</li>
              </ul>
            </Box>
          </Grid>
          <Grid item xs={12} md={6}>
            <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>
              <Typography variant="h6" gutterBottom color="error">
                Limitations
              </Typography>
              <ul>
                <li>Requires substantially more data than lower-moment statistics for reliable estimation.</li>
                <li>Highly sensitive to outliers and estimation error.</li>
                <li>Computationally intensive for large portfolios.</li>
                <li>Challenging to implement in standard optimization frameworks.</li>
                <li>Benefits may be marginal in some markets compared to simpler three-moment approaches.</li>
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
            <li>Fang, H., & Lai, T. Y. (1997). <em>Co-kurtosis and capital asset pricing</em>. <em>The Financial Review</em>, 32(2), 293-307.</li>
            <li>Dittmar, R. F. (2002). <em>Nonlinear pricing kernels, kurtosis preference, and evidence from the cross section of equity returns</em>. <em>Journal of Finance</em>, 57(1), 369-403.</li>
            <li>Christoffersen, P., Feunou, B., Jacobs, K., & Turnbull, S. (2021). <em>Option-Based Estimation of the Price of Coskewness and Cokurtosis Risk</em>. <em>Journal of Financial and Quantitative Analysis</em>, 56(1), 65-91.</li>
            <li>Jondeau, E., & Rockinger, M. (2006). <em>Optimal portfolio allocation under higher moments</em>. <em>European Financial Management</em>, 12(1), 29-55.</li>
            <li>Martellini, L., & Ziemann, V. (2010). <em>Improved estimates of higher-order comoments and implications for portfolio selection</em>. <em>The Review of Financial Studies</em>, 23(4), 1467-1502.</li>
            <li>Guidolin, M., & Timmermann, A. (2008). <em>International asset allocation under regime switching, skew, and kurtosis preferences</em>. <em>The Review of Financial Studies</em>, 21(2), 889-935.</li>
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
                  Coskewness
                </Typography>
                <Typography variant="body2" paragraph>
                  A third-moment measure that quantifies how an asset's returns interact with squared market returns.
                </Typography>
                <Link href="/docs/coskewness" passHref>
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
                  Value-at-Risk
                </Typography>
                <Typography variant="body2" paragraph>
                  A statistical technique used to measure the level of financial risk within a portfolio over a specific time frame.
                </Typography>
                <Link href="/docs/value-at-risk" passHref>
                  <Button variant="contained" color="primary">
                    Learn More
                  </Button>
                </Link>
              </Box>
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <Box sx={{ p: 2, border: '1px solid #e0e0e0', borderRadius: 1, height: '100%' }}>
                <Typography variant="h6" gutterBottom>
                  Conditional Value-at-Risk
                </Typography>
                <Typography variant="body2" paragraph>
                  A risk measure that quantifies the expected loss in the worst-case scenarios beyond the VaR threshold.
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
      </Container>
    </>
  );
};

export default CokurtosisPage; 