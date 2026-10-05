/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later
 * Model discovery follows SpatialSlideshow's MIT-licensed research; see NOTICE. */
#import "ModelResolver.h"
#import <dlfcn.h>

@interface UAFAssetSetManager : NSObject
+ (id)sharedManager;
- (id)retrieveAssetSet:(NSString*)name usages:(NSDictionary*)usages;
@end
@interface UAFAssetSet : NSObject
- (NSDictionary*)assets;
@end
@interface UAFAsset : NSObject
- (NSURL*)location;
@end

static NSString* const AssetRoot = @"/System/Library/AssetsV2/";

static NSDictionary* modelsInRoots(NSArray<NSURL*>* roots) {
    NSMutableSet<NSString*>*joint = [NSMutableSet new], *fov = [NSMutableSet new];
    NSFileManager* fm = NSFileManager.defaultManager;
    for (NSURL* root in roots) {
        if (![root.URLByResolvingSymlinksInPath.path hasPrefix:AssetRoot])
            continue;
        for (NSURL* model in [fm contentsOfDirectoryAtURL:root includingPropertiesForKeys:nil options:0 error:nil]) {
            if (![model.pathExtension isEqual:@"mlmodelc"] ||
                ![fm isReadableFileAtPath:[model URLByAppendingPathComponent:@"coremldata.bin"].path] ||
                ![fm isReadableFileAtPath:[model URLByAppendingPathComponent:@"model.specialization.bundle"].path])
                continue;
            if ([model.lastPathComponent containsString:@"joint_predictor"])
                [joint addObject:model.path];
            if ([model.lastPathComponent containsString:@"fov_"])
                [fov addObject:model.path];
        }
    }
    return joint.count == 1 && fov.count == 1 ? @{@"joint" : joint.anyObject, @"fov" : fov.anyObject} : @{};
}

NSDictionary<NSString*, NSString*>* LFSReframeModels(void) {
    // Resolve Apple's active model set first. OS updates can leave several asset
    // revisions installed; neither timestamps nor directory order select a model.
    @try {
        void* framework = dlopen("/System/Library/PrivateFrameworks/UnifiedAssetFramework.framework/UnifiedAssetFramework", RTLD_NOW | RTLD_LOCAL);
        Class cls = framework ? NSClassFromString(@"UAFAssetSetManager") : Nil;
        id manager = [cls respondsToSelector:@selector(sharedManager)] ? [cls sharedManager] : nil;
        if ([manager respondsToSelector:@selector(retrieveAssetSet:usages:)]) {
            NSDictionary* usages = @{@"com.apple.photos.spatialphotosrelive.main.generic" : @"ENABLED",
                                     @"com.apple.photos.spatialphotosrelive.fov.main.generic" : @"ENABLED"};
            UAFAssetSet* set = [manager retrieveAssetSet:@"com.apple.MobileAsset.UAF.Photos.SpatialPhotosRelive" usages:usages];
            NSMutableArray* roots = [NSMutableArray new];
            for (NSString* name in usages) {
                UAFAsset* asset = [set assets][name];
                NSURL* location = [asset respondsToSelector:@selector(location)] ? asset.location : nil;
                if (location.isFileURL)
                    [roots addObject:location];
            }
            NSDictionary* active = modelsInRoots(roots);
            if (active.count == 2)
                return active;
        }
    } @catch (NSException* exception) { /* Changed private interface: fail closed or unique fallback. */
    }
    NSURL* directory = [NSURL fileURLWithPath:[AssetRoot stringByAppendingString:@"com_apple_MobileAsset_UAF_Photos_SpatialPhotosRelive/purpose_auto"]];
    NSMutableArray* roots = [NSMutableArray new];
    for (NSURL* asset in [NSFileManager.defaultManager contentsOfDirectoryAtURL:directory includingPropertiesForKeys:nil options:0 error:nil]) {
        if ([asset.pathExtension isEqual:@"asset"])
            [roots addObject:[asset URLByAppendingPathComponent:@".AssetData"]];
    }
    return modelsInRoots(roots);
}
