const fs = require("fs");
const path = require("path");
const hre = require("hardhat");

async function deployOne(name) {
  const Factory = await hre.ethers.getContractFactory(name);
  const contract = await Factory.deploy();
  await contract.waitForDeployment();
  const artifact = await hre.artifacts.readArtifact(name);
  return {
    address: await contract.getAddress(),
    abi: artifact.abi
  };
}

async function main() {
  const deployments = {};
  for (const name of [
    "FederationRegistry",
    "TrainingLedger",
    "ContributionToken",
    "MisbehaviorRegistry"
  ]) {
    deployments[name] = await deployOne(name);
    console.log(`${name}: ${deployments[name].address}`);
  }

  const outputDir = path.join(__dirname, "..", "artifacts", "chain");
  fs.mkdirSync(outputDir, { recursive: true });
  fs.writeFileSync(
    path.join(outputDir, "deployment.json"),
    JSON.stringify(deployments, null, 2)
  );
  console.log(`Deployment written to ${path.join(outputDir, "deployment.json")}`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
