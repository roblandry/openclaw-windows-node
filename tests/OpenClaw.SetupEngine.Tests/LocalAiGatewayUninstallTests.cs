using OpenClaw.Connection.LocalAi;
using OpenClaw.Shared.Inference;
using OpenClaw.Shared.Inference.Catalog;
using OpenClaw.TestSupport;
using System.Text;
using System.Text.Json;

namespace OpenClaw.SetupEngine.Tests;

public sealed class LocalAiGatewayUninstallTests
{
    [Fact]
    public async Task FreshProcessModelReplacement_TransitionsPriorRouteAndFinalizesReceipt()
    {
        using var temp = new TempDirectory("local-ai-gateway-replacement-resume-");
        LocalAiResolvedInstall previous = await SaveManifestAsync(temp.Path, "openai/gpt-5");
        LocalAiInstallManifest replacementManifest = previous.Manifest with
        {
            ModelCatalogId = LocalModelCatalog.Qwen27BModelId,
            ModelAlias = LocalModelCatalog.Qwen27BModelId,
            GatewayFallbackModel = null,
        };
        var paths = new LocalAiPaths(temp.Path);
        (LocalAiResolvedInstall replacement, LocalAiModelReplacement recovery) =
            await SaveReplacementAsync(paths, previous, replacementManifest);
        var commands = new GatewayStateCommandRunner(
            LocalAiGatewayProviderDefinition.BuildProviderJson(previous),
            JsonSerializer.Serialize(LocalAiGatewayProviderDefinition.BuildPrimaryModel(previous)));
        SetupContext context = CreateContext(temp.Path, commands);
        context.Config.LocalAi.Enabled = true;
        context.LocalAiResolvedInstall = replacement;
        context.ReplacedLocalAiInstall = previous;
        context.LocalAiModelReplacement = recovery;
        context.LocalAiEligibility = LocalInferenceEligibility.Evaluate(
            CreateQualifiedHardware(),
            LocalModelCatalog.Qwen27BModelId);

        StepResult result = await new ConfigureLocalAiGatewayStep()
            .ExecuteAsync(context, CancellationToken.None);

        Assert.Equal(StepOutcome.Success, result.Outcome);
        Assert.Equal(
            LocalAiGatewayProviderDefinition.BuildProviderJson(context.LocalAiResolvedInstall!),
            commands.ProviderJson);
        Assert.Equal("openai/gpt-5", context.LocalAiResolvedInstall!.Manifest.GatewayFallbackModel);
        Assert.NotNull((await new LocalAiManifestStore(paths).LoadAsync())!.Manifest.ModelReplacement);

        await new ConfigureLocalAiGatewayStep().RollbackAsync(context, CancellationToken.None);

        Assert.Equal(LocalAiGatewayProviderDefinition.BuildProviderJson(previous), commands.ProviderJson);
        Assert.Equal(
            JsonSerializer.Serialize(LocalAiGatewayProviderDefinition.BuildPrimaryModel(previous)),
            commands.PrimaryJson);
        Assert.NotNull((await new LocalAiManifestStore(paths).LoadAsync())!.Manifest.ModelReplacement);
    }

    [Fact]
    public async Task FreshProcessModelReplacement_AcceptsPreviouslyRecordedAutomaticPort()
    {
        using var temp = new TempDirectory("local-ai-gateway-replacement-port-resume-");
        LocalAiResolvedInstall previous = await SaveManifestAsync(temp.Path, "openai/gpt-5");
        LocalAiInstallManifest replacementManifest = previous.Manifest with
        {
            ModelCatalogId = LocalModelCatalog.Qwen27BModelId,
            ModelAlias = LocalModelCatalog.Qwen27BModelId,
            Endpoint = "http://127.0.0.1:28766/v1",
            GatewayFallbackModel = "openai/gpt-5",
        };
        const string publishedEndpoint = "http://127.0.0.1:28765/v1";
        var paths = new LocalAiPaths(temp.Path);
        (LocalAiResolvedInstall replacement, LocalAiModelReplacement recovery) =
            await SaveReplacementAsync(paths, previous, replacementManifest, publishedEndpoint);
        LocalAiResolvedInstall publishedReplacement = new LocalAiManifestStore(paths)
            .ResolveAndValidate(replacement.Manifest with { Endpoint = publishedEndpoint });
        var commands = new GatewayStateCommandRunner(
            LocalAiGatewayProviderDefinition.BuildProviderJson(publishedReplacement),
            JsonSerializer.Serialize(LocalAiGatewayProviderDefinition.BuildPrimaryModel(publishedReplacement)));
        SetupContext context = CreateContext(temp.Path, commands);
        context.Config.LocalAi.Enabled = true;
        context.LocalAiResolvedInstall = replacement;
        context.ReplacedLocalAiInstall = previous;
        context.LocalAiModelReplacement = recovery;
        context.LocalAiEligibility = LocalInferenceEligibility.Evaluate(
            CreateQualifiedHardware(),
            LocalModelCatalog.Qwen27BModelId);

        StepResult result = await new ConfigureLocalAiGatewayStep()
            .ExecuteAsync(context, CancellationToken.None);

        Assert.Equal(StepOutcome.Success, result.Outcome);
        Assert.Equal(LocalAiGatewayProviderDefinition.BuildProviderJson(replacement), commands.ProviderJson);
        LocalAiModelReplacement persisted =
            (await new LocalAiManifestStore(paths).LoadAsync())!.Manifest.ModelReplacement!;
        Assert.Contains(replacement.Endpoint!.AbsoluteUri, persisted.GatewayEndpoints);
    }

    [Fact]
    public async Task RejectedReplacementRoute_DoesNotReleaseRollbackProtection()
    {
        using var temp = new TempDirectory("local-ai-gateway-replacement-rejected-");
        LocalAiResolvedInstall previous = await SaveManifestAsync(temp.Path, "openai/gpt-5");
        LocalAiInstallManifest replacementManifest = previous.Manifest with
        {
            ModelCatalogId = LocalModelCatalog.Qwen27BModelId,
            ModelAlias = LocalModelCatalog.Qwen27BModelId,
            GatewayFallbackModel = "openai/gpt-5",
        };
        var paths = new LocalAiPaths(temp.Path);
        (LocalAiResolvedInstall replacement, LocalAiModelReplacement recovery) =
            await SaveReplacementAsync(paths, previous, replacementManifest);
        string modifiedProvider = LocalAiGatewayProviderDefinition.BuildProviderJson(replacement).Replace(
            "\"apiKey\":\"llama-local\"",
            "\"apiKey\":\"modified\"",
            StringComparison.Ordinal);
        var commands = new GatewayStateCommandRunner(
            modifiedProvider,
            JsonSerializer.Serialize(LocalAiGatewayProviderDefinition.BuildPrimaryModel(replacement)));
        SetupContext context = CreateContext(temp.Path, commands);
        context.Config.LocalAi.Enabled = true;
        context.LocalAiResolvedInstall = replacement;
        context.ReplacedLocalAiInstall = previous;
        context.LocalAiModelReplacement = recovery;
        context.LocalAiModelReplacementRollbackBlocked = true;
        context.LocalAiManifestCreatedThisRun = true;
        context.LocalAiEligibility = LocalInferenceEligibility.Evaluate(
            CreateQualifiedHardware(),
            LocalModelCatalog.Qwen27BModelId);
        var step = new ConfigureLocalAiGatewayStep();

        StepResult result = await step.ExecuteAsync(context, CancellationToken.None);
        await step.RollbackAsync(context, CancellationToken.None);
        await new PersistLocalAiManifestStep().RollbackAsync(context, CancellationToken.None);

        Assert.Equal(StepOutcome.Failed, result.Outcome);
        Assert.True(context.LocalAiModelReplacementRollbackBlocked);
        Assert.NotNull((await new LocalAiManifestStore(paths).LoadAsync())!.Manifest.ModelReplacement);
        Assert.Equal(
            replacementManifest.ModelCatalogId,
            (await new LocalAiManifestStore(paths).LoadAsync())!.Manifest.ModelCatalogId);
        Assert.Equal(modifiedProvider, commands.ProviderJson);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task FreshProcessUninstall_RemovesRecordedReplacementRoutes(bool useReplacementRoute)
    {
        using var temp = new TempDirectory("local-ai-gateway-replacement-port-uninstall-");
        LocalAiResolvedInstall previous = await SaveManifestAsync(temp.Path, "openai/gpt-5");
        LocalAiInstallManifest replacementManifest = previous.Manifest with
        {
            ModelCatalogId = LocalModelCatalog.Qwen27BModelId,
            ModelAlias = LocalModelCatalog.Qwen27BModelId,
            Endpoint = "http://127.0.0.1:28766/v1",
            GatewayFallbackModel = "openai/gpt-5",
        };
        const string publishedEndpoint = "http://127.0.0.1:28765/v1";
        var paths = new LocalAiPaths(temp.Path);
        (LocalAiResolvedInstall replacement, _) =
            await SaveReplacementAsync(paths, previous, replacementManifest, publishedEndpoint);
        LocalAiResolvedInstall publishedReplacement = new LocalAiManifestStore(paths)
            .ResolveAndValidate(replacement.Manifest with { Endpoint = publishedEndpoint });
        LocalAiResolvedInstall route = useReplacementRoute ? publishedReplacement : previous;
        var commands = new GatewayStateCommandRunner(
            LocalAiGatewayProviderDefinition.BuildProviderJson(route),
            JsonSerializer.Serialize(LocalAiGatewayProviderDefinition.BuildPrimaryModel(route)));
        SetupContext context = CreateContext(temp.Path, commands);
        context.IsUninstalling = true;

        await new ConfigureLocalAiGatewayStep().RollbackAsync(context, CancellationToken.None);

        Assert.Null(commands.ProviderJson);
        Assert.Equal(JsonSerializer.Serialize("openai/gpt-5"), commands.PrimaryJson);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task ModelReplacement_FailedGatewayRollbackPreservesDurableReplacement(
        bool throwDuringRestore)
    {
        using var temp = new TempDirectory("local-ai-gateway-replacement-rollback-");
        LocalAiResolvedInstall previous = await SaveManifestAsync(temp.Path, "openai/gpt-5");
        LocalAiInstallManifest replacementManifest = previous.Manifest with
        {
            ModelCatalogId = LocalModelCatalog.Qwen27BModelId,
            ModelAlias = LocalModelCatalog.Qwen27BModelId,
            GatewayFallbackModel = "openai/gpt-5",
        };
        var paths = new LocalAiPaths(temp.Path);
        (LocalAiResolvedInstall replacement, LocalAiModelReplacement recovery) =
            await SaveReplacementAsync(paths, previous, replacementManifest);
        var commands = new GatewayStateCommandRunner(
            LocalAiGatewayProviderDefinition.BuildProviderJson(previous),
            JsonSerializer.Serialize(LocalAiGatewayProviderDefinition.BuildPrimaryModel(previous)));
        SetupContext context = CreateContext(temp.Path, commands);
        context.Config.LocalAi.Enabled = true;
        context.LocalAiResolvedInstall = replacement;
        context.ReplacedLocalAiInstall = previous;
        context.LocalAiModelReplacement = recovery;
        context.LocalAiManifestCreatedThisRun = true;
        context.LocalAiEligibility = LocalInferenceEligibility.Evaluate(
            CreateQualifiedHardware(),
            LocalModelCatalog.Qwen27BModelId);
        var step = new ConfigureLocalAiGatewayStep();
        Assert.Equal(
            StepOutcome.Success,
            (await step.ExecuteAsync(context, CancellationToken.None)).Outcome);
        commands.FailRestore = !throwDuringRestore;
        commands.ThrowRestore = throwDuringRestore;

        Assert.NotNull(await Record.ExceptionAsync(() =>
            step.RollbackAsync(context, CancellationToken.None)));
        await new PersistLocalAiManifestStep().RollbackAsync(context, CancellationToken.None);

        Assert.True(context.LocalAiModelReplacementRollbackBlocked);
        Assert.NotNull((await new LocalAiManifestStore(paths).LoadAsync())!.Manifest.ModelReplacement);
        Assert.Equal(
            replacementManifest.ModelCatalogId,
            (await new LocalAiManifestStore(paths).LoadAsync())!.Manifest.ModelCatalogId);
        Assert.Equal(
            LocalAiGatewayProviderDefinition.BuildProviderJson(replacement),
            commands.ProviderJson);
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public async Task FreshProcessUninstall_PreservesMixedReplacementRoute(bool previousProvider)
    {
        using var temp = new TempDirectory("local-ai-gateway-replacement-mixed-");
        LocalAiResolvedInstall previous = await SaveManifestAsync(temp.Path, "openai/gpt-5");
        LocalAiInstallManifest replacementManifest = previous.Manifest with
        {
            ModelCatalogId = LocalModelCatalog.Qwen27BModelId,
            ModelAlias = LocalModelCatalog.Qwen27BModelId,
            GatewayFallbackModel = "openai/gpt-5",
        };
        var paths = new LocalAiPaths(temp.Path);
        (LocalAiResolvedInstall replacement, _) =
            await SaveReplacementAsync(paths, previous, replacementManifest);
        string provider = LocalAiGatewayProviderDefinition.BuildProviderJson(
            previousProvider ? previous : replacement);
        string primary = JsonSerializer.Serialize(LocalAiGatewayProviderDefinition.BuildPrimaryModel(
            previousProvider ? replacement : previous));
        var commands = new GatewayStateCommandRunner(provider, primary);
        SetupContext context = CreateContext(temp.Path, commands);
        context.IsUninstalling = true;

        await Assert.ThrowsAsync<InvalidDataException>(() =>
            new ConfigureLocalAiGatewayStep().RollbackAsync(context, CancellationToken.None));

        Assert.Equal(provider, commands.ProviderJson);
        Assert.Equal(primary, commands.PrimaryJson);
    }

    [Fact]
    public async Task FreshProcessUninstall_RemovesExactManagedProviderAndPrimary()
    {
        using var temp = new TempDirectory("local-ai-gateway-uninstall-");
        LocalAiResolvedInstall install = await SaveManifestAsync(temp.Path);
        string provider = LocalAiGatewayProviderDefinition.BuildProviderJson(install);
        string primary = JsonSerializer.Serialize(
            LocalAiGatewayProviderDefinition.BuildPrimaryModel(install));
        var commands = new GatewayStateCommandRunner(provider, primary);
        SetupContext context = CreateContext(temp.Path, commands);
        context.IsUninstalling = true;

        await new ConfigureLocalAiGatewayStep().RollbackAsync(context, CancellationToken.None);

        Assert.Null(commands.ProviderJson);
        Assert.Null(commands.PrimaryJson);
        Assert.Contains(commands.WslCalls, command =>
            command.Contains("LOCAL_AI_GATEWAY_UNSET", StringComparison.Ordinal));
    }

    [Fact]
    public async Task FreshProcessUninstall_AcceptsCliRedactedManagedApiKey()
    {
        using var temp = new TempDirectory("local-ai-gateway-uninstall-");
        LocalAiResolvedInstall install = await SaveManifestAsync(temp.Path);
        string provider = LocalAiGatewayProviderDefinition.BuildProviderJson(install).Replace(
            "\"api\":\"openai-completions\",\"apiKey\":\"llama-local\"",
            $"\"apiKey\":\"{LocalAiGatewayProviderDefinition.CliRedactedApiKey}\",\"api\":\"openai-completions\"",
            StringComparison.Ordinal);
        string primary = JsonSerializer.Serialize(
            LocalAiGatewayProviderDefinition.BuildPrimaryModel(install));
        var commands = new GatewayStateCommandRunner(provider, primary);
        SetupContext context = CreateContext(temp.Path, commands);
        context.IsUninstalling = true;

        await new ConfigureLocalAiGatewayStep().RollbackAsync(context, CancellationToken.None);

        Assert.Null(commands.ProviderJson);
        Assert.Null(commands.PrimaryJson);
    }

    [Fact]
    public async Task FreshProcessUninstall_RestoresRecordedFallbackPrimary()
    {
        using var temp = new TempDirectory("local-ai-gateway-uninstall-");
        LocalAiResolvedInstall install = await SaveManifestAsync(temp.Path, "openai/gpt-5");
        string provider = LocalAiGatewayProviderDefinition.BuildProviderJson(install);
        string primary = JsonSerializer.Serialize(
            LocalAiGatewayProviderDefinition.BuildPrimaryModel(install));
        var commands = new GatewayStateCommandRunner(provider, primary);
        SetupContext context = CreateContext(temp.Path, commands);
        context.IsUninstalling = true;

        await new ConfigureLocalAiGatewayStep().RollbackAsync(context, CancellationToken.None);

        Assert.Null(commands.ProviderJson);
        Assert.Equal(JsonSerializer.Serialize("openai/gpt-5"), commands.PrimaryJson);
        Assert.Contains(commands.WslCalls, command =>
            command.Contains("LOCAL_AI_PRIMARY_RESTORED", StringComparison.Ordinal));
    }

    [Fact]
    public async Task FreshProcessUninstall_PreservesDriftAndFailsClosed()
    {
        using var temp = new TempDirectory("local-ai-gateway-uninstall-");
        LocalAiResolvedInstall install = await SaveManifestAsync(temp.Path);
        string expectedProvider = LocalAiGatewayProviderDefinition.BuildProviderJson(install);
        string driftedProvider = expectedProvider.Replace(
            "http://127.0.0.1:28765/v1",
            "http://127.0.0.1:39876/v1",
            StringComparison.Ordinal);
        string primary = JsonSerializer.Serialize(
            LocalAiGatewayProviderDefinition.BuildPrimaryModel(install));
        var commands = new GatewayStateCommandRunner(driftedProvider, primary);
        SetupContext context = CreateContext(temp.Path, commands);
        context.IsUninstalling = true;

        InvalidDataException error = await Assert.ThrowsAsync<InvalidDataException>(() =>
            new ConfigureLocalAiGatewayStep().RollbackAsync(context, CancellationToken.None));

        Assert.Contains("preserving", error.Message, StringComparison.OrdinalIgnoreCase);
        Assert.Equal(driftedProvider, commands.ProviderJson);
        Assert.Equal(primary, commands.PrimaryJson);
        Assert.DoesNotContain(commands.WslCalls, command =>
            command.Contains("LOCAL_AI_GATEWAY_UNSET", StringComparison.Ordinal));
    }

    [Fact]
    public async Task FreshProcessUninstall_PreservesStateWhenSnapshotFails()
    {
        using var temp = new TempDirectory("local-ai-gateway-uninstall-");
        LocalAiResolvedInstall install = await SaveManifestAsync(temp.Path);
        string provider = LocalAiGatewayProviderDefinition.BuildProviderJson(install);
        string primary = JsonSerializer.Serialize(
            LocalAiGatewayProviderDefinition.BuildPrimaryModel(install));
        var commands = new GatewayStateCommandRunner(provider, primary) { FailCapture = true };
        SetupContext context = CreateContext(temp.Path, commands);
        context.IsUninstalling = true;

        await Assert.ThrowsAsync<IOException>(() =>
            new ConfigureLocalAiGatewayStep().RollbackAsync(context, CancellationToken.None));

        Assert.Equal(provider, commands.ProviderJson);
        Assert.Equal(primary, commands.PrimaryJson);
        Assert.DoesNotContain(commands.WslCalls, command =>
            command.Contains("LOCAL_AI_GATEWAY_UNSET", StringComparison.Ordinal));
    }

    private static SetupContext CreateContext(string localDataDirectory, ICommandRunner commands)
    {
        var config = new SetupConfig();
        var logger = new SetupLogger(filePath: null);
        return new SetupContext(
            config,
            logger,
            new TransactionJournal(filePath: null),
            commands,
            CancellationToken.None,
            localDataDir: localDataDirectory);
    }

    private static async Task<(LocalAiResolvedInstall Install, LocalAiModelReplacement Recovery)>
        SaveReplacementAsync(
            LocalAiPaths paths,
            LocalAiResolvedInstall previous,
            LocalAiInstallManifest replacement,
            params string[] gatewayEndpoints)
    {
        var recovery = new LocalAiModelReplacement
        {
            PreviousManifest = previous.Manifest,
            RouterPresetExisted = false,
            GatewayEndpoints = [.. gatewayEndpoints],
        };
        var store = new LocalAiManifestStore(paths);
        await store.SaveAsync(replacement with { ModelReplacement = recovery });
        return ((await store.LoadAsync())!, recovery);
    }

    private static HostHardwareInfo CreateQualifiedHardware() => new(
        System.Runtime.InteropServices.Architecture.Arm64,
        128L * 1024 * 1024 * 1024,
        100L * 1024 * 1024 * 1024,
        [
            new GpuInfo(
                GpuVendor.Nvidia,
                "NVIDIA RTX Spark N1X (6144-core Blackwell RTX GPU)",
                GpuVisibleMemoryBytes: 64L * 1024 * 1024 * 1024,
                FreeGpuVisibleMemoryBytes: 60L * 1024 * 1024 * 1024,
                DriverVersion: "616.00",
                CudaMajorVersion: 13,
                StableId: "GPU-SPARK"),
        ],
        VulkanAvailable: false);

    private static async Task<LocalAiResolvedInstall> SaveManifestAsync(
        string localDataDirectory,
        string? fallbackModel = null)
    {
        var paths = new LocalAiPaths(localDataDirectory);
        const string revision = "5bc3e238d916f48a861bac2f8a1990a0e9b7e98d";
        var manifest = new LocalAiInstallManifest
        {
            EngineVersion = "b10488",
            Architecture = "arm64",
            HardwareProfileId = "rtx-spark-n1x",
            RuntimeId = "b10488-cuda13-arm64",
            ModelCatalogId = LocalModelCatalog.Qwen35BModelId,
            SelectedGpuId = "GPU-SPARK",
            ExecutablePath = Path.Combine("engines", "llama-b10488", "llama-server.exe"),
            RuntimeAssets =
            [
                new LocalAiAssetReceipt
                {
                    FileName = "llama-runtime.zip",
                    SourceUrl = "https://github.com/ggml-org/llama.cpp/releases/download/b10488/llama-runtime.zip",
                    SizeBytes = 1,
                    Sha256 = new string('a', 64),
                },
            ],
            ModelPath = Path.Combine("models", "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
            ModelId = $"unsloth/Qwen3.6-35B-A3B-MTP-GGUF@{revision}",
            ModelAlias = LocalModelCatalog.Qwen35BModelId,
            ModelAsset = new LocalAiAssetReceipt
            {
                FileName = "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
                SourceUrl = $"https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/resolve/{revision}/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf?download=true",
                SizeBytes = 1,
                Sha256 = new string('b', 64),
            },
            RequestedPort = 0,
            Endpoint = "http://127.0.0.1:28765/v1",
            GatewayFallbackModel = fallbackModel,
            ContextLength = LocalModelCatalog.NativeContextTokens,
        };
        var store = new LocalAiManifestStore(paths);
        await store.SaveAsync(manifest);
        return (await store.LoadAsync())!;
    }

    private sealed class GatewayStateCommandRunner(
        string? providerJson,
        string? primaryJson) : ICommandRunner
    {
        private const string ProviderMarker = "OPENCLAW_LOCAL_AI_PROVIDER_B64=";
        private const string PrimaryMarker = "OPENCLAW_LOCAL_AI_PRIMARY_B64=";

        public string? ProviderJson { get; private set; } = providerJson;
        public string? PrimaryJson { get; private set; } = primaryJson;
        public bool FailCapture { get; init; }
        public bool FailRestore { get; set; }
        public bool ThrowRestore { get; set; }
        public List<string> WslCalls { get; } = [];

        public Task<CommandResult> RunAsync(
            string executable,
            string[] arguments,
            TimeSpan timeout,
            IReadOnlyDictionary<string, string>? environment = null,
            string? workingDirectory = null,
            string? stdinInput = null,
            CancellationToken ct = default,
            Stream? stdinStream = null) => throw new NotSupportedException();

        public Task<CommandResult> RunInWslAsync(
            string distroName,
            string command,
            TimeSpan timeout,
            IReadOnlyDictionary<string, string>? environment = null,
            CancellationToken ct = default,
            string? user = null,
            bool inputViaStdin = false)
        {
            ct.ThrowIfCancellationRequested();
            WslCalls.Add(command);
            if (command.Contains("LOCAL_AI_PRIMARY_RESTORED", StringComparison.Ordinal))
            {
                string encoded = Assert.Single(environment!).Value;
                string batch = Encoding.UTF8.GetString(Convert.FromBase64String(encoded));
                using JsonDocument document = JsonDocument.Parse(batch);
                PrimaryJson = document.RootElement[0].GetProperty("value").GetRawText();
                return Task.FromResult(new CommandResult(
                    0,
                    "LOCAL_AI_PRIMARY_RESTORED",
                    "",
                    TimeSpan.Zero,
                    TimedOut: false));
            }
            if (command.Contains("LOCAL_AI_GATEWAY_CONFIGURED", StringComparison.Ordinal) ||
                command.Contains("LOCAL_AI_GATEWAY_RESTORED", StringComparison.Ordinal))
            {
                if (ThrowRestore && command.Contains("LOCAL_AI_GATEWAY_RESTORED", StringComparison.Ordinal))
                    throw new OperationCanceledException("restore timed out");
                if (FailRestore && command.Contains("LOCAL_AI_GATEWAY_RESTORED", StringComparison.Ordinal))
                {
                    return Task.FromResult(new CommandResult(
                        1,
                        "",
                        "restore failed",
                        TimeSpan.Zero,
                        TimedOut: false));
                }
                string encoded = Assert.Single(environment!).Value;
                string batch = Encoding.UTF8.GetString(Convert.FromBase64String(encoded));
                using JsonDocument document = JsonDocument.Parse(batch);
                foreach (JsonElement operation in document.RootElement.EnumerateArray())
                {
                    string path = operation.GetProperty("path").GetString()!;
                    string value = operation.GetProperty("value").GetRawText();
                    if (path == LocalAiGatewayProviderDefinition.ProviderPath)
                        ProviderJson = value;
                    else if (path == LocalAiGatewayProviderDefinition.PrimaryModelPath)
                        PrimaryJson = value;
                }
                string marker = command.Contains("LOCAL_AI_GATEWAY_CONFIGURED", StringComparison.Ordinal)
                    ? "LOCAL_AI_GATEWAY_CONFIGURED"
                    : "LOCAL_AI_GATEWAY_RESTORED";
                return Task.FromResult(new CommandResult(
                    0,
                    marker,
                    "",
                    TimeSpan.Zero,
                    TimedOut: false));
            }
            if (command.Contains("LOCAL_AI_GATEWAY_UNSET", StringComparison.Ordinal))
            {
                if (command.Contains(LocalAiGatewayProviderDefinition.PrimaryModelPath, StringComparison.Ordinal))
                    PrimaryJson = null;
                if (command.Contains(LocalAiGatewayProviderDefinition.ProviderPath, StringComparison.Ordinal))
                    ProviderJson = null;
                return Task.FromResult(new CommandResult(
                    0,
                    "LOCAL_AI_GATEWAY_UNSET",
                    "",
                    TimeSpan.Zero,
                    TimedOut: false));
            }
            if (FailCapture)
            {
                return Task.FromResult(new CommandResult(
                    1,
                    "",
                    "openclaw config get failed",
                    TimeSpan.Zero,
                    TimedOut: false));
            }

            string stdout =
                ProviderMarker + EncodeOrMissing(ProviderJson) + Environment.NewLine +
                PrimaryMarker + EncodeOrMissing(PrimaryJson) + Environment.NewLine;
            return Task.FromResult(new CommandResult(0, stdout, "", TimeSpan.Zero, TimedOut: false));
        }

        private static string EncodeOrMissing(string? value) => value is null
            ? "MISSING"
            : Convert.ToBase64String(Encoding.UTF8.GetBytes(value));
    }
}
