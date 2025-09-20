/** @type {import('next-sitemap').IConfig} */
module.exports = {
  siteUrl:
    process.env.SITE_URL || "https://indportfoliooptimization.vercel.app",
  generateRobotsTxt: true,
  // Force a single sitemap file (no index)
  generateIndexSitemap: false,
  sitemapSize: 50000,
  changefreq: "weekly",
  priority: 0.7,
  exclude: ["/api/*", "/_next/*"],
  transform: async (config, path) => {
    const high = ["/", "/dividend", "/about", "/docs"];
    return {
      loc: path,
      changefreq: high.includes(path) ? "daily" : config.changefreq,
      priority: high.includes(path) ? 0.9 : config.priority,
      lastmod: new Date().toISOString(),
    };
  },
  robotsTxtOptions: {
    policies:
      process.env.NODE_ENV === "production"
        ? [{ userAgent: "*", allow: "/" }]
        : [{ userAgent: "*", disallow: "/" }],
  },
};
